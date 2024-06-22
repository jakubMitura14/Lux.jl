@testitem "xlogx & xlogy" setup=[SharedTestSetup] tags=[:helpers] begin
    using Lux: xlogx, xlogy
    using ForwardDiff, Zygote

    @test iszero(xlogx(0))
    @test isnan(xlogx(NaN))
    @test xlogx(2) ≈ 2.0 * log(2.0)

    ∂x1 = ForwardDiff.derivative(xlogx, 2.0)
    ∂x2 = Zygote.gradient(xlogx, 2.0)[1]
    @test ∂x1 ≈ ∂x2

    @inferred xlogx(2)
    @inferred xlogx(0)
    @jet xlogx(2)

    @test iszero(xlogy(0, 1))
    @test isnan(xlogy(NaN, 1))
    @test isnan(xlogy(1, NaN))
    @test isnan(xlogy(NaN, NaN))
    @test xlogy(2, 3) ≈ 2.0 * log(3.0)

    ∂x1 = ForwardDiff.derivative(Base.Fix2(xlogy, 3.0), 2.0)
    ∂y1 = ForwardDiff.derivative(Base.Fix1(xlogy, 2.0), 3.0)
    ∂x2, ∂y2 = Zygote.gradient(xlogy, 2.0, 3.0)
    @test ∂x1 ≈ ∂x2
    @test ∂y1 ≈ ∂y2

    @inferred xlogy(2, 3)
    @inferred xlogy(0, 1)
    @jet xlogy(2, 3)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = rand(10) |> aType
        __f = sum ∘ Broadcast.BroadcastFunction(xlogx)
        @eval @test_gradients $__f $x gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3

        y = rand(10) |> aType
        __f = sum ∘ Broadcast.BroadcastFunction(xlogy)
        @eval @test_gradients $__f $x $y gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
    end
end

@testitem "Regression Loss" setup=[SharedTestSetup] tags=[:helpers] begin
    using Zygote

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        y = [1.0, 1.0, 0.0, 0.0] |> aType
        ŷ = [0.9, 0.1, 0.1, 0.9] |> aType

        loss_res_map = Dict(
            "MSE" => (0.1^2 + 0.9^2) / 2, "MAE" => (0.1 + 0.9) / 2, "Huber" => 0.205)

        @testset "$(loss)" for (loss, loss_res) in loss_res_map
            loss_mean = eval(Symbol(loss * "Loss"))()
            loss_sum = eval(Symbol(loss * "Loss"))(; agg=sum)
            loss_sum2 = eval(Symbol(loss * "Loss"))(; agg=(args...) -> sum(args...))

            @test loss_mean(ŷ, y) ≈ loss_res
            @test loss_sum(ŷ, y) ≈ loss_res * 4
            @test loss_sum2(ŷ, y) ≈ loss_res * 4

            @inferred Zygote.gradient(loss_mean, ŷ, y)

            @jet loss_mean(ŷ, y)
            @jet loss_sum(ŷ, y)

            __f = Base.Fix2(loss_mean, y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "MSLE" begin
            y = [123.0, 456.0, 789.0] |> aType
            ŷ = [345.0, 332.0, 789.0] |> aType

            @test MSLELoss()(ŷ, y) ≈ 0.38813985859136585

            @jet MSLELoss()(ŷ, y)

            @test_broken @inferred Zygote.gradient(MSLELoss(), ŷ, y)

            __f = Base.Fix2(MSLELoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end
    end
end

@testitem "Classification Loss" setup=[SharedTestSetup] tags=[:helpers] begin
    using OneHotArrays, Zygote

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        y = onehotbatch([1, 1, 0, 0], 0:1) |> dev
        y_smoothed = Lux.__label_smoothing(0.1, y, Float32)

        ŷ = [0.1 0.9; 0.9 0.1; 0.9 0.1; 0.1 0.9]' |> dev
        v = log(0.1 / 0.9)
        logŷ = [v 0.0; 0.0 v; 0.0 v; v 0.0]' |> dev
        lossvalue = 1.203972804325936
        lossvalue_smoothed = 1.2039728043259348

        yl = onehotbatch([1], 0:1) |> dev
        sf = 0.1
        yls = [sf (1 - sf)]' |> dev
        ylp = [0.9 0.1]' |> dev
        logylp = [0.0 v]' |> dev

        ya = onehotbatch([1, 1, 1, 0, 0], 0:1) |> dev
        ya_smoothed = Lux.__label_smoothing(2sf, ya, Float32)
        y_same = Float32.(ya)
        y_sim = y_same .* (1 - 2 * sf) .+ sf
        y_dis = copy(y_sim)
        y_dis[1, :], y_dis[2, :] = y_dis[2, :], y_dis[1, :]

        @testset "CrossEntropyLoss" begin
            celoss = CrossEntropyLoss()

            @test celoss([0.1, 0.0, 0.9] |> aType, [0.1, 0.0, 0.9] |> aType) ≈
                  celoss([0.1, 0.9] |> aType, [0.1, 0.9] |> aType)

            @test celoss(ŷ, y) ≈ lossvalue
            @test celoss(ŷ, y_smoothed) ≈ lossvalue_smoothed

            celoss_smooth = CrossEntropyLoss(; label_smoothing=0.1)
            @test celoss_smooth(ŷ, y) ≈ lossvalue_smoothed

            celoss_smooth2 = CrossEntropyLoss(; label_smoothing=2sf)
            @test celoss_smooth2(ylp, yl) ≈ sum(-yls .* log.(ylp))

            @test celoss(ylp, yl) ≈ sum(-yl .* log.(ylp))

            @test iszero(CrossEntropyLoss(; epsilon=0)(y_same, ya))

            @test celoss(y_sim, ya) < celoss_smooth(y_sim, ya)
            @test celoss(y_dis, ya) > celoss_smooth(y_dis, ya)

            @jet celoss(ŷ, y)
            @jet celoss_smooth(ŷ, y)

            @inferred Zygote.gradient(celoss, ŷ, y)

            __f = Base.Fix2(celoss, y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "Logit CrossEntropyLoss" begin
            logitceloss = CrossEntropyLoss(; logits=Val(true))

            @test logitceloss(logŷ, y) ≈ lossvalue
            @test logitceloss(logylp, yl) ≈ sum(-yl .* log.(softmax(logylp)))

            logitceloss_smooth = CrossEntropyLoss(; logits=Val(true), label_smoothing=0.1)

            @test logitceloss(logŷ, y_smoothed) ≈ lossvalue_smoothed
            @test logitceloss_smooth(logŷ, y) ≈ lossvalue_smoothed

            logitceloss_smooth2 = CrossEntropyLoss(; logits=Val(true), label_smoothing=2sf)
            @test logitceloss_smooth2(logylp, yl) ≈ sum(-yls .* log.(softmax(logylp)))

            @jet logitceloss(logŷ, y)
            @jet logitceloss_smooth(logŷ, y)

            @inferred Zygote.gradient(logitceloss, logŷ, y)

            __f = Base.Fix2(logitceloss, y)
            @eval @test_gradients $__f $logŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        logŷ, y = randn(3) |> aType, rand(3) |> aType
        yls = y .* (1 - 2sf) .+ sf

        @testset "BinaryCrossEntropyLoss" begin
            bceloss = BinaryCrossEntropyLoss()
            bceloss_smooth = BinaryCrossEntropyLoss(; label_smoothing=2sf, epsilon=0)

            @test bceloss_smooth(σ.(logŷ), y) ≈
                  -mean(yls .* log.(σ.(logŷ)) .+ (1 .- yls) .* log.(1 .- σ.(logŷ)))

            @test bceloss(σ.(logŷ), y) ≈
                  mean(-y .* log.(σ.(logŷ)) .- (1 .- y) .* log.(1 .- σ.(logŷ)))

            @test bceloss(σ.(logŷ), y) ≈ mean(-y .* log.(σ.(logŷ) .+ eps.(σ.(logŷ))) -
                       (1 .- y) .* log.(1 .- σ.(logŷ) .+ eps.(σ.(logŷ))))

            @test bceloss([0.1, 0.2, 0.9] |> aType, 1) ≈
                  -mean(log, [0.1, 0.2, 0.9] |> aType)  # constant label

            @jet bceloss(σ.(logŷ), y)
            @jet bceloss_smooth(σ.(logŷ), y)

            @inferred Zygote.gradient(bceloss, σ.(logŷ), y)

            __f = Base.Fix2(bceloss, y)
            σlogŷ = σ.(logŷ)
            @eval @test_gradients $__f $σlogŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "Logit BinaryCrossEntropyLoss" begin
            logitbceloss = BinaryCrossEntropyLoss(; logits=Val(true))
            logitbceloss_smooth = BinaryCrossEntropyLoss(;
                logits=Val(true), label_smoothing=2sf, epsilon=0)

            @test logitbceloss_smooth(logŷ, y) ≈
                  -mean(yls .* log.(sigmoid(logŷ)) .+
                        (1 .- yls) .* log.(1 .- sigmoid(logŷ)))

            @test logitbceloss(logŷ, y) ≈
                  mean(-y .* log.(sigmoid(logŷ)) .- (1 .- y) .* log.(1 .- sigmoid(logŷ)))

            @jet logitbceloss(logŷ, y)
            @jet logitbceloss_smooth(logŷ, y)

            @inferred Zygote.gradient(logitbceloss, logŷ, y)

            __f = Base.Fix2(logitbceloss, y)
            @eval @test_gradients $__f $logŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "BinaryFocalLoss" begin
            y = [0 1 0
                 1 0 1] |> aType
            ŷ = [0.268941 0.5 0.268941
                 0.731059 0.5 0.731059] |> aType

            y1 = [1 0
                  0 1] |> aType
            ŷ1 = [0.6 0.3
                  0.4 0.7] |> aType

            @test BinaryFocalLoss()(ŷ, y) ≈ 0.0728675615927385
            @test BinaryFocalLoss()(ŷ1, y1) ≈ 0.05691642237852222
            @test BinaryFocalLoss(; gamma=0)(ŷ, y) ≈ Lux.CrossEntropyLoss()(ŷ, y)

            @jet BinaryFocalLoss()(ŷ, y)

            if ongpu
                @test_broken @inferred Zygote.gradient(BinaryFocalLoss(), ŷ, y)
            else
                @inferred Zygote.gradient(BinaryFocalLoss(), ŷ, y)
            end

            __f = Base.Fix2(BinaryFocalLoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "FocalLoss" begin
            y = [1 0 0 0 1
                 0 1 0 1 0
                 0 0 1 0 0] |> aType
            ŷ = softmax(reshape(-7:7, 3, 5) .* 1.0f0) |> aType
            y1 = [1 0
                  0 0
                  0 1] |> aType
            ŷ1 = [0.4 0.2
                  0.5 0.5
                  0.1 0.3] |> aType

            @test FocalLoss()(ŷ, y) ≈ 1.1277571935622628
            @test FocalLoss()(ŷ1, y1) ≈ 0.45990566879720157
            @test FocalLoss(; gamma=0)(ŷ, y) ≈ CrossEntropyLoss()(ŷ, y)

            @jet FocalLoss()(ŷ, y)

            if ongpu
                @test_broken @inferred Zygote.gradient(FocalLoss(), ŷ, y)
            else
                @inferred Zygote.gradient(FocalLoss(), ŷ, y)
            end

            __f = Base.Fix2(FocalLoss(), y)
            # FD will lead to out of domain errors
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu skip_finite_differences=true
        end
    end
end

@testitem "Other Losses" setup=[SharedTestSetup] tags=[:helpers] begin
    using Zygote

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "KLDivergenceLoss" begin
            y = [1 2 3] |> aType
            ŷ = [4.0 5.0 6.0] |> aType

            @test KLDivergenceLoss()([0.1, 0.0, 0.9] |> aType, [0.1, 0.0, 0.9] |> aType) ≈
                  KLDivergenceLoss()([0.1, 0.9] |> aType, [0.1, 0.9] |> aType)
            @test KLDivergenceLoss()(ŷ, y) ≈ -1.7661057888493457
            @test KLDivergenceLoss()(y, y) ≈ 0

            @jet KLDivergenceLoss()(ŷ, y)
            @inferred Zygote.gradient(KLDivergenceLoss(), ŷ, y)

            __f = Base.Fix2(KLDivergenceLoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "HingeLoss" begin
            y = [1, 2, 3, 4] |> aType
            ŷ = [5.0, 6.0, 7.0, 8.0] |> aType

            @test Lux.HingeLoss()(ŷ, y) ≈ 0
            @test Lux.HingeLoss()(y, 0.5 .* y) ≈ 0.125

            @jet Lux.HingeLoss()(ŷ, y)
            @inferred Zygote.gradient(Lux.HingeLoss(), ŷ, y)

            __f = Base.Fix2(Lux.HingeLoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "SquaredHingeLoss" begin
            y = [1, 2, 3, 4] |> aType
            ŷ = [5.0, 6.0, 7.0, 8.0] |> aType

            @test SquaredHingeLoss()(ŷ, y) ≈ 0
            @test SquaredHingeLoss()(y, 0.5 .* y) ≈ 0.0625

            @jet SquaredHingeLoss()(ŷ, y)
            @inferred Zygote.gradient(SquaredHingeLoss(), ŷ, y)

            __f = Base.Fix2(SquaredHingeLoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "PoissonLoss" begin
            y = [0.1, 0.2, 0.3] |> aType
            ŷ = [0.4, 0.5, 0.6] |> aType

            @test Lux.PoissonLoss()(ŷ, y) ≈ 0.6278353988097339
            @test Lux.PoissonLoss()(y, y) ≈ 0.5044459776946685

            @jet Lux.PoissonLoss()(ŷ, y)
            @test_broken @inferred Zygote.gradient(Lux.PoissonLoss(), ŷ, y)

            __f = Base.Fix2(Lux.PoissonLoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "DiceCoeffLoss" begin
            y = [1.0, 0.5, 0.3, 2.4] |> aType
            ŷ = [0.0, 1.4, 0.5, 1.2] |> aType

            @test DiceCoeffLoss()(ŷ, y) ≈ 0.2799999999999999
            @test DiceCoeffLoss()(y, y) ≈ 0.0

            @jet DiceCoeffLoss()(ŷ, y)
            @test_broken @inferred Zygote.gradient(DiceCoeffLoss(), ŷ, y)

            __f = Base.Fix2(DiceCoeffLoss(), y)
            @eval @test_gradients $__f $ŷ gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3 skip_tracker=$ongpu
        end

        @testset "Siamese Contrastive Loss" begin
            y = [1 0
                 0 0
                 0 1] |> aType
            ŷ = [0.4 0.2
                 0.5 0.5
                 0.1 0.3] |> aType
            y1 = [1 0 0 0 1
                  0 1 0 1 0
                  0 0 1 0 0] |> aType
            ŷ1 = softmax(reshape(-7:7, 3, 5) .* 1.0f0) |> aType
            y2 = [1
                  0
                  0
                  1
                  1] |> aType
            ŷ2 = [0.6
                  0.4
                  0.1
                  0.2
                  0.7] |> aType

            @test SiameseContrastiveLoss()(ŷ, y) ≈ 0.2333333333333333
            @test SiameseContrastiveLoss(; margin=0.5f0)(ŷ, y) ≈ 0.10000000000000002
            @test SiameseContrastiveLoss(; margin=1.5f0)(ŷ, y) ≈ 0.5333333333333333
            @test SiameseContrastiveLoss()(ŷ1, y1) ≈ 0.32554644f0
            @test SiameseContrastiveLoss(; margin=0.5f0)(ŷ1, y1) ≈ 0.16271012f0
            @test SiameseContrastiveLoss(; margin=1.5f0)(ŷ1, y1) ≈ 0.6532292f0
            @test SiameseContrastiveLoss(; margin=1)(ŷ, y) ≈ SiameseContrastiveLoss()(ŷ, y)
            @test SiameseContrastiveLoss()(y, y) ≈ 0.0
            @test SiameseContrastiveLoss()(y1, y1) ≈ 0.0
            @test SiameseContrastiveLoss(; margin=0)(ŷ, y) ≈ 0.09166666666666667
            @test SiameseContrastiveLoss(; margin=0)(ŷ1, y1) ≈ 0.13161165f0
            @test SiameseContrastiveLoss()(ŷ2, y2) ≈ 0.21200000000000005
            @test SiameseContrastiveLoss()(ŷ2, ŷ2) ≈ 0.18800000000000003

            @jet SiameseContrastiveLoss()(ŷ, y)

            @test_throws ArgumentError SiameseContrastiveLoss(; margin=-0.5)
            @test_throws ArgumentError SiameseContrastiveLoss(; margin=-1)
        end
    end
end

@testitem "Losses: Error Checks and Misc" setup=[SharedTestSetup] tags=[:helpers] begin
    @testset "Size Checks" begin
        @test_throws DimensionMismatch MSELoss()([1, 2], [1, 2, 3])
    end

    @testset "No Aggregation" begin
        @test MSELoss(; agg=nothing)([1, 3], [3, 1]) == [4, 4]
    end

    @testset "Scalar Loss" begin
        @test MSELoss(; agg=sum)(1.0, 1.0) == 0.0
        @test MSELoss(; agg=sum)(2.0, 0.0) == 4.0
        @test MSLELoss(; agg=sum)(2.0, 0.0) ≈ Lux.__msle_loss(2.0, 0.0, nothing)
    end
end
