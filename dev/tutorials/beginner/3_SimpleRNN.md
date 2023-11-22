


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
((lstm_cell = (weight_i = Float32[-0.8617533 -0.41677052; -0.25580457 -0.7539321; 0.80954915 0.88440406; -0.057783905 0.07662625; -0.1463586 -0.6221104; 0.28884095 0.5549227; -0.8348161 0.07171448; 0.0803929 0.04625299; -0.04195189 0.076544724; -0.6620829 0.4536679; 1.1405008 -0.019875314; -0.012466605 0.0878238; -0.106648035 0.24767965; -0.84829545 0.32500717; -0.12993304 -0.2720008; 1.2134793 0.09380071; 0.6803301 -0.6291039; 0.13039081 -0.34585458; 0.47471303 -0.32568222; -0.55503184 0.5166102; 0.5849803 -0.83659106; 0.1002567 0.29128766; 0.86819214 -0.6398685; 0.9524021 0.6192799; -1.2415792 -0.113640316; 0.48274642 0.64357847; 1.0558242 0.6783089; -0.4615123 -0.17153655; 0.72597814 -0.7091838; -0.33858582 0.7449389; -0.2156039 0.6095402; 0.6067963 -0.31679067], weight_h = Float32[-0.5043255 -0.053005867 0.2911059 -0.2643495 0.2973671 0.013736772 -0.68587655 0.5497464; -0.6517398 0.22684903 -0.064812616 0.6367885 0.38471928 0.31210423 0.12969157 -0.05195829; 0.01687694 0.062493943 -0.03501042 0.56037194 -0.7383357 0.3173276 -0.6362865 -0.17248976; 0.03923037 -0.2724334 0.88468957 -0.097437695 0.91090304 0.086772636 0.28087094 0.9070471; -0.3674187 0.47204494 0.7733308 0.29652634 0.40099692 0.64186215 -0.33713526 -0.07989856; -0.061855637 -0.33639392 0.33209488 -0.09563379 -0.29065362 -0.21911396 -0.49851656 0.03616463; -0.78763866 0.37133923 0.68919426 0.50516784 -0.80730534 0.6401674 0.02606312 -0.63308775; 0.68720883 0.30815312 1.0453873 -1.3188369 0.82136947 0.16771586 -0.71646196 1.1414094; -0.15182754 0.47618532 0.41673112 -0.5078983 0.4577772 0.68278146 0.18121329 0.6344868; -0.4197891 -0.18575649 -0.38041657 -0.010568483 0.2436573 0.0016102373 -0.7466923 0.7354012; -0.1494076 0.5535998 -0.05421091 0.5263279 -0.5162067 0.29555255 -0.27013028 -0.3789174; 0.05947519 0.8786751 0.28664094 -0.11175067 0.82848316 0.6225743 0.39484 0.49920437; 0.6527316 0.48583236 0.37301156 -0.070336156 0.92012876 0.18588305 -0.6152982 0.44113573; -0.10456182 0.4425631 0.08606647 0.083714776 0.6093061 -0.010554291 -0.135274 -0.41344678; -0.8288377 0.32775584 0.0812549 -0.4000698 -0.25818768 0.799369 -0.3694751 0.40506306; -0.6483447 0.7873353 0.41064602 1.0701712 -0.0794501 0.81560796 -0.105278105 0.77358276; 0.14795944 -0.63867944 -0.049442977 -0.4359854 -0.35820764 -0.2204092 -0.16257729 0.17986065; -0.43883315 0.22904783 0.19846241 0.71455836 0.40100887 -0.35266814 -0.36425772 0.6714059; -0.5544028 0.70954174 0.095020264 -0.43144748 -0.30050778 0.7410424 0.049608957 0.6013759; -0.7754902 0.34051493 -0.16609499 0.50251806 -0.69610274 -0.13885427 -0.2001059 -0.19625302; -0.27415168 -0.66648227 0.39984262 -0.6407527 -0.17312247 0.2639429 0.25008172 -0.12119833; -0.7267168 -0.42355877 0.53314054 -0.4767562 -0.14699854 0.028628977 -0.4417327 0.21652445; 0.48588693 -0.19124642 -0.66186434 0.28280395 0.33704925 0.06784646 -0.52198243 -0.40900248; -0.41779426 0.35601 0.22980358 0.37330166 -0.34292763 -0.19328831 -0.74524444 0.098729946; -0.64752823 0.556079 0.2724997 -0.46058607 0.59603673 0.44023356 -0.9548471 0.86509705; -0.6088967 0.27799964 0.067266084 0.43062013 -0.26187688 0.5700759 -0.4185386 -0.3269615; -0.8205347 -0.27152136 0.4681848 -0.27265763 0.2357604 0.013715707 -0.89123505 0.74524146; 0.28864008 0.27301747 1.0179033 -0.9103988 0.24901475 0.18963283 -1.1114506 0.47032285; -0.33365187 0.27752835 0.7730488 -0.43003818 0.44820505 0.9234165 -1.1816493 0.69302005; 0.42545602 0.40620998 -0.6866022 0.6227715 -1.1633481 -0.36393645 -0.16758424 -0.053689092; -0.42821768 0.51800203 -0.25396135 0.7021314 -0.8164483 -0.118060544 0.08204642 -0.09161369; -0.5424867 0.7609592 0.70051277 -0.5550647 0.62358785 0.06561831 -0.49398795 0.646886], bias = Float32[0.29172358; 0.2819535; 0.13594101; 0.32213706; 0.34365782; 0.11864625; 0.011040268; 0.97796494; 0.38185188; 0.15590319; 0.008879655; 0.35594612; 0.42991698; 0.0674976; -0.012377037; 0.32245374; 0.844677; 1.1296866; 1.1569366; 0.7696779; 0.68789953; 1.23973; 0.63969845; 0.9166867; 0.47723415; 0.035210565; 0.3007132; 0.61871785; 0.62723845; 0.027168963; -0.11868853; 0.882217;;]), classifier = (weight = Float32[-1.4401857 0.76206046 1.244072 1.2645471 -0.93446857 0.120631516 -0.26257035 1.2245182], bias = Float32[-0.5851478;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

