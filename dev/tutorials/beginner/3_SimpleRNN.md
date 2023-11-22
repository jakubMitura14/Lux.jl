


<a id='Training-a-Simple-LSTM'></a>

# Training a Simple LSTM


In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:


1. Create custom Lux models.
2. Become familiar with the Lux recurrent neural network API.
3. Training using Optimisers.jl and Zygote.jl.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Random, Statistics
```


<a id='Dataset'></a>

## Dataset


We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq*len × batch*size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.


```julia
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(d[1][:, (sequence_length + 1):end], :,
        sequence_length, 1) for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end
```


```
get_dataloaders (generic function with 1 method)
```


<a id='Creating-a-Classifier'></a>

## Creating a Classifier


We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.


We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.


To understand more about container layers, please look at [Container Layer](http://lux.csail.mit.edu/stable/manual/interface/#container-layer).


```julia
struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```


We won't define the model from scratch but rather use the [`Lux.LSTMCell`](../../api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](../../api/Lux/layers#Lux.Dense).


```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(LSTMCell(in_dims => hidden_dims),
        Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##292".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(x::AbstractArray{T, 3}, ps::NamedTuple,
        st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    # Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end
```


<a id='Defining-Accuracy,-Loss-and-Optimiser'></a>

## Defining Accuracy, Loss and Optimiser


Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.


```julia
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.


```julia
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end
```


```
create_optimiser (generic function with 1 method)
```


<a id='Training-the-Model'></a>

## Training the Model


```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        # Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end

    return (ps, st) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
((lstm_cell = (weight_i = Float32[-0.86368877 -0.42126417; -0.25239405 -0.7542533; 0.8062372 0.8874734; -0.06335414 0.0752275; -0.14468043 -0.6243675; 0.2861344 0.54703134; -0.83571243 0.07348565; 0.07771998 0.048117496; -0.03983902 0.08249759; -0.6645707 0.45168117; 1.1784908 -0.01851327; -0.01937384 0.08659066; -0.10746292 0.24651651; -0.85460734 0.31730792; -0.12713517 -0.27159107; 1.2154623 0.09921211; 0.6836417 -0.6269524; 0.12700287 -0.34277046; 0.4581544 -0.32463038; -0.5490141 0.5180442; 0.58348227 -0.83790255; 0.09910738 0.29153758; 0.87199134 -0.63809836; 0.9608279 0.6201881; -1.2332717 -0.10351864; 0.47470948 0.6440952; 1.0544851 0.6926736; -0.47155744 -0.16499588; 0.72751564 -0.7089779; -0.3418151 0.73257744; -0.21636099 0.609653; 0.62421626 -0.3103003], weight_h = Float32[-0.5125219 -0.054959636 0.32695392 -0.2539642 0.29904485 0.008979678 -0.6785086 0.550553; -0.66873574 0.21790296 -0.05707456 0.6220266 0.38085976 0.30876338 0.13010432 -0.05604556; 0.012897101 0.064262316 -0.021734618 0.5596552 -0.74040437 0.31863552 -0.63628435 -0.16034491; 0.041268796 -0.2657941 0.88612247 -0.09666846 0.91174483 0.091877505 0.28415045 0.9073969; -0.3687199 0.47099456 0.7808286 0.30137363 0.4022276 0.6400314 -0.33850455 -0.079199426; -0.054706924 -0.3363579 0.3299695 -0.09653668 -0.2857456 -0.21768838 -0.47874016 0.044393215; -0.7886732 0.36926702 0.6942465 0.51416594 -0.8102244 0.63696843 0.0236263 -0.6305784; 0.6913351 0.30503553 1.0616003 -1.3142962 0.8127836 0.17147608 -0.7238634 1.1424466; -0.14899564 0.47267982 0.41689205 -0.5032114 0.4609254 0.678213 0.18431485 0.63812333; -0.4187501 -0.185873 -0.38373095 -0.007944238 0.24088559 -0.0028377413 -0.76707083 0.7056572; -0.14943244 0.5493339 -0.029131627 0.5276836 -0.5218503 0.29448378 -0.2685258 -0.38067538; 0.06102478 0.88114816 0.2876181 -0.10842022 0.8296744 0.6325045 0.39555922 0.49962303; 0.6528614 0.48446408 0.37240097 -0.06846707 0.92014533 0.1771401 -0.63640773 0.44100407; -0.09744928 0.44647327 0.08383103 0.093084686 0.6127273 -0.012120447 -0.14256802 -0.41063902; -0.8315931 0.32699522 0.08060666 -0.399782 -0.25917056 0.8054009 -0.37364686 0.40463623; -0.6454302 0.78712463 0.4157913 1.06664 -0.08023825 0.8152468 -0.10175618 0.76653624; 0.14415348 -0.64185125 -0.050187286 -0.43776777 -0.35725155 -0.21871044 -0.16543874 0.17595407; -0.44191316 0.23170114 0.19391797 0.7177478 0.40108588 -0.35860205 -0.36489913 0.656169; -0.5547097 0.7041882 0.09212311 -0.42986196 -0.32985103 0.7417043 0.049942255 0.6007012; -0.77756757 0.33895522 -0.16907737 0.50484705 -0.69773215 -0.1288761 -0.20151141 -0.19616872; -0.26972958 -0.66364324 0.3955135 -0.6391786 -0.17580548 0.2650121 0.25354812 -0.11464707; -0.71741796 -0.43623072 0.5308536 -0.4900717 -0.13093033 0.033418134 -0.44259146 0.22417292; 0.48533666 -0.19960143 -0.6671951 0.27607003 0.3375646 0.070458695 -0.5233951 -0.41066012; -0.41903615 0.35585845 0.22945741 0.3746143 -0.31417593 -0.19029336 -0.7478036 0.07053586; -0.6479357 0.5521588 0.26977897 -0.44810414 0.59665704 0.4350887 -0.9497224 0.8683098; -0.6055642 0.2666389 0.07253266 0.42806154 -0.26621485 0.56177956 -0.4161112 -0.3235669; -0.81837964 -0.26954874 0.46458185 -0.290586 0.26266223 0.013650255 -0.89297664 0.7609121; 0.27473477 0.27003726 1.0180749 -0.90137875 0.2466482 0.18817733 -1.1163853 0.5000809; -0.34738213 0.27634543 0.77271193 -0.417287 0.44829106 0.91256356 -1.1875882 0.6973739; 0.43781835 0.3974848 -0.69048303 0.629759 -1.190393 -0.37295148 -0.15660547 -0.06455339; -0.43059695 0.51528364 -0.25616753 0.7291204 -0.8165623 -0.128247 0.08510598 -0.089794986; -0.5476614 0.7538533 0.69086343 -0.55340433 0.6337023 0.059619416 -0.49469504 0.6452073], bias = Float32[0.2918584; 0.27333334; 0.14647736; 0.32521552; 0.3441088; 0.1212113; 0.0095298; 0.97510946; 0.38300326; 0.15707636; 0.00761107; 0.35737705; 0.43002534; 0.071555994; -0.014290465; 0.32412472; 0.8419321; 1.1327264; 1.1578323; 0.7681148; 0.6897984; 1.2317806; 0.6362484; 0.91706604; 0.47606492; 0.04358058; 0.31204593; 0.6186246; 0.6277995; 0.019485766; -0.11906848; 0.8819519;;]), classifier = (weight = Float32[-1.4373767 0.7729674 1.2393067 1.259413 -0.9339941 0.114324115 -0.2615106 1.2285286], bias = Float32[-0.60918546;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
```


<a id='Saving-the-Model'></a>

## Saving the Model


We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model


```julia
@save "trained_model.jld2" {compress = true} ps_trained st_trained
```


Let's try loading the model


```julia
@load "trained_model.jld2" ps_trained st_trained
```


```
2-element Vector{Symbol}:
 :ps_trained
 :st_trained
```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

