include("DataAndGenerate.jl")
include("Convolution.jl")
include("Activations.jl")
include("AdamUpdate.jl")
include("PixelShuffle.jl")

function singleLayerTest()
    conv = Convolution(3, 3, 3, 3)
    step = Convolution(3, 3, 3, 3)

    c = 16


    output = zeros(Float32, (c, 3, 32, 32))
    deltas = zeros(Float32, (c, 3, 32, 32))

    stepsize = 0.001
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    mwprev = 0
    mbprev = 0
    vwprev = 0
    vbprev = 0
    mw = []
    mb = []
    vw = []
    vb = []
    for j = 1:10
        data[:, :, :, :] = data[shuffle(1:end), :, :, :]
        @time for i = 1:c:50000

            output = Convolve!(data[i:i+c-1,:,:,:], output, conv)
            deltas = output - data[i:i+c-1,:,:,:]
            value = size(deltas, 1) * size(deltas, 2) * size(deltas, 3) * size(deltas, 4)
            println(sum(abs.(deltas))./value)

            step = WeightStep!(data[i:i+c-1, :, :, :], deltas, step)

            mw = (b1 .* mwprev) .+ (1-b1) .* step.weights
            mb = (b1 .* mbprev) .+ (1-b1) .* step.bias

            vw = (b2 .* vwprev) .+ (1-b2) .* step.weights .* step.weights
            vb = (b2 .* vbprev) .+ (1-b2) .* step.bias .* step.bias


            conv.weights = conv.weights - stepsize .* (mw) ./ (sqrt.(vw) .+ eps)
            conv.bias = conv.bias - stepsize .* (mb) ./ (sqrt.(vb) .+ eps)

            mwprev = mw
            mbprev = mb
            vwprev = vw
            vbprev = vb

        end
        stepsize /= 2
    end

    output = Convolve!(data[1:c, :, :, :], output, conv)
    return output
end



function linearLayersTest()
    f = 8
    conv1 = Convolution(3, 3, 3, f)
    conv2 = Convolution(3, 3, f, f)
    conv3 = Convolution(3, 3, f, 3)
    step1 = Convolution(3, 3, 3, f)
    step2 = Convolution(3, 3, f, f)
    step3 = Convolution(3, 3, f, 3)

    c = 16

    output1 = zeros(Float32, (c, f, 32, 32))
    output2 = zeros(Float32, (c, f, 32, 32))
    output3 = zeros(Float32, (c, 3, 32, 32))
    deltas1 = zeros(Float32, (c, f, 32, 32))
    deltas2 = zeros(Float32, (c, f, 32, 32))
    deltas3 = zeros(Float32, (c, 3, 32, 32))
    derivatives1 = ones(Float32, (c, f, 32, 32))
    derivatives2 = ones(Float32, (c, f, 32, 32))

    stepsize = 0.0005
    b1 = 0.9
    b2 = 0.999
    eps = 1e-6
    mw1prev = 0
    mb1prev = 0
    vw1prev = 0
    vb1prev = 0
    mw1 = []
    mb1 = []
    vw1 = []
    vb1 = []
    mw2prev = 0
    mb2prev = 0
    vw2prev = 0
    vb2prev = 0
    mw2 = []
    mb2 = []
    vw2 = []
    vb2 = []
    mw3prev = 0
    mb3prev = 0
    vw3prev = 0
    vb3prev = 0
    mw3 = []
    mb3 = []
    vw3 = []
    vb3 = []
    #data[:, :, :, :] = data[shuffle(1:end), :, :, :]
    
    for j = 1:1
        noisy, truth = generate(50000)
        noisy = truth
        @time for i = 1:c:50000
            output1 = Convolve!(noisy[i:i+c-1, :, :, :], output1, conv1)
            output2 = Convolve!(output1, output2, conv2)
            output3 = Convolve!(output2, output3, conv3)

            deltas3 = (output3 - truth[i:i+c-1, :, :, :])
            value = size(deltas3, 1) * size(deltas3, 2) * size(deltas3, 3) * size(deltas3, 4)
            println(sum(abs.(deltas3))./value)

            step3 = WeightStep!(output2, deltas3, step3)
            #step3.weights = gradClip(step3.weights, 5)
            #step3.bias = gradClip(step3.bias, 5)

            mw3 = (b1 .* mw3prev) .+ (1-b1) .* step3.weights
            mb3 = (b1 .* mb3prev) .+ (1-b1) .* step3.bias
            vw3 = (b2 .* vw3prev) .+ (1-b2) .* step3.weights .* step3.weights
            vb3 = (b2 .* vb3prev) .+ (1-b2) .* step3.bias .* step3.bias

            deltas2 = BackConvolve!(deltas3, deltas2, derivatives2, conv3)
            step2 = WeightStep!(output1, deltas2, step2)
            #step2.weights = gradClip(step2.weights, 5)
            #step2.bias = gradClip(step2.bias, 5)

            mw2 = (b1 .* mw2prev) .+ (1-b1) .* step2.weights
            mb2 = (b1 .* mb2prev) .+ (1-b1) .* step2.bias

            vw2 = (b2 .* vw2prev) .+ (1-b2) .* step2.weights .* step2.weights
            vb2 = (b2 .* vb2prev) .+ (1-b2) .* step2.bias .* step2.bias

            deltas1 = BackConvolve!(deltas2, deltas1, derivatives1, conv2)
            step1 = WeightStep!(noisy[i:i+c-1, :, :, :], deltas1, step1)
            #step1.weights = gradClip(step1.weights, 5)
            #step1.bias = gradClip(step1.bias, 5)

            mw1 = (b1 .* mw1prev) .+ (1-b1) .* step1.weights
            mb1 = (b1 .* mb1prev) .+ (1-b1) .* step1.bias

            vw1 = (b2 .* vw1prev) .+ (1-b2) .* step1.weights .* step1.weights
            vb1 = (b2 .* vb1prev) .+ (1-b2) .* step1.bias .* step1.bias
            


            conv1.weights = conv1.weights - stepsize .* ((mw1) ./ (sqrt.(vw1) .+ eps))
            conv1.bias = conv1.bias - stepsize .* ((mb1) ./ (sqrt.(vb1) .+ eps))
            conv2.weights = conv2.weights - stepsize .* ((mw2) ./ (sqrt.(vw2) .+ eps))
            conv2.bias = conv2.bias - stepsize .* ((mb2) ./ (sqrt.(vb2) .+ eps))
            conv3.weights = conv3.weights - stepsize .* ((mw3) ./ (sqrt.(vw3) .+ eps))
            conv3.bias = conv3.bias - stepsize .* ((mb3) ./ (sqrt.(vb3) .+ eps))
            mw1prev = mw1
            mw2prev = mw2
            mw3prev = mw3
            mb1prev = mb1
            mb2prev = mb2
            mb3prev = mb3
            vw1prev = vw1
            vw2prev = vw2
            vw3prev = vw3
            vb1prev = vb1
            vb2prev = vb2
            vb3prev = vb3
        end
    end

    output1 = Convolve!(data[1:c, :, :, :], output1, conv1)
    output2 = Convolve!(output1, output2, conv2)
    output3 = Convolve!(output2, output3, conv3)
    return output3
end

function simpleNeuralNetTest()
    f = 8
    conv1 = Convolution(3, 3, 3, f)
    conv2 = Convolution(3, 3, f, f)
    conv3 = Convolution(3, 3, f, 3)
    step1 = Convolution(3, 3, 3, f)
    step2 = Convolution(3, 3, f, f)
    step3 = Convolution(3, 3, f, 3)

    c = 32

    output1 = zeros(Float32, (c, f, 32, 32))
    output2 = zeros(Float32, (c, f, 32, 32))
    output3 = zeros(Float32, (c, 3, 32, 32))
    deltas1 = zeros(Float32, (c, f, 32, 32))
    deltas2 = zeros(Float32, (c, f, 32, 32))
    deltas3 = zeros(Float32, (c, 3, 32, 32))
    derivatives1 = ones(Float32, (c, f, 32, 32))
    derivatives2 = ones(Float32, (c, f, 32, 32))

    stepsize = 0.001
    b1 = 0.9
    b2 = 0.999
    eps = 1e-6
    mw1prev = 0
    mb1prev = 0
    vw1prev = 0
    vb1prev = 0
    mw1 = []
    mb1 = []
    vw1 = []
    vb1 = []
    mw2prev = 0
    mb2prev = 0
    vw2prev = 0
    vb2prev = 0
    mw2 = []
    mb2 = []
    vw2 = []
    vb2 = []
    mw3prev = 0
    mb3prev = 0
    vw3prev = 0
    vb3prev = 0
    mw3 = []
    mb3 = []
    vw3 = []
    vb3 = []
    #data[:, :, :, :] = data[shuffle(1:end), :, :, :]
    
    for j = 1:10
        noisy, truth = generate(50000+c)
        #noisy = truth
        @time for i = 1:c:50000
            output1 = Convolve!(noisy[i:i+c-1, :, :, :], output1, conv1)
            derivatives1 = dReLU!(derivatives1, output1)
            output1 = ReLU!(output1)
            output2 = Convolve!(output1, output2, conv2)
            derivatives2 = dReLU!(derivatives2, output2)
            output2 = ReLU!(output2)
            output3 = Convolve!(output2, output3, conv3)

            output3 = max.(min.(output3, 1), 0)

            deltas3 = (output3 - truth[i:i+c-1, :, :, :])
            value = size(deltas3, 1) * size(deltas3, 2) * size(deltas3, 3) * size(deltas3, 4)
            println(sum(abs.(deltas3))./value)

            step3 = WeightStep!(output2, deltas3, step3)
            #step3.weights = gradClip(step3.weights, 5)
            #step3.bias = gradClip(step3.bias, 5)

            mw3 = (b1 .* mw3prev) .+ (1-b1) .* step3.weights
            mb3 = (b1 .* mb3prev) .+ (1-b1) .* step3.bias
            vw3 = (b2 .* vw3prev) .+ (1-b2) .* step3.weights .* step3.weights
            vb3 = (b2 .* vb3prev) .+ (1-b2) .* step3.bias .* step3.bias

            deltas2 = BackConvolve!(deltas3, deltas2, derivatives2, conv3)
            step2 = WeightStep!(output1, deltas2, step2)
            #step2.weights = gradClip(step2.weights, 5)
            #step2.bias = gradClip(step2.bias, 5)

            mw2 = (b1 .* mw2prev) .+ (1-b1) .* step2.weights
            mb2 = (b1 .* mb2prev) .+ (1-b1) .* step2.bias

            vw2 = (b2 .* vw2prev) .+ (1-b2) .* step2.weights .* step2.weights
            vb2 = (b2 .* vb2prev) .+ (1-b2) .* step2.bias .* step2.bias

            deltas1 = BackConvolve!(deltas2, deltas1, derivatives1, conv2)
            step1 = WeightStep!(noisy[i:i+c-1, :, :, :], deltas1, step1)
            #step1.weights = gradClip(step1.weights, 5)
            #step1.bias = gradClip(step1.bias, 5)

            mw1 = (b1 .* mw1prev) .+ (1-b1) .* step1.weights
            mb1 = (b1 .* mb1prev) .+ (1-b1) .* step1.bias

            vw1 = (b2 .* vw1prev) .+ (1-b2) .* step1.weights .* step1.weights
            vb1 = (b2 .* vb1prev) .+ (1-b2) .* step1.bias .* step1.bias
            


            conv1.weights = conv1.weights - stepsize .* ((mw1) ./ (sqrt.(vw1) .+ eps))
            conv1.bias = conv1.bias - stepsize .* ((mb1) ./ (sqrt.(vb1) .+ eps))
            conv2.weights = conv2.weights - stepsize .* ((mw2) ./ (sqrt.(vw2) .+ eps))
            conv2.bias = conv2.bias - stepsize .* ((mb2) ./ (sqrt.(vb2) .+ eps))
            conv3.weights = conv3.weights - stepsize .* ((mw3) ./ (sqrt.(vw3) .+ eps))
            conv3.bias = conv3.bias - stepsize .* ((mb3) ./ (sqrt.(vb3) .+ eps))
            mw1prev = mw1
            mw2prev = mw2
            mw3prev = mw3
            mb1prev = mb1
            mb2prev = mb2
            mb3prev = mb3
            vw1prev = vw1
            vw2prev = vw2
            vw3prev = vw3
            vb1prev = vb1
            vb2prev = vb2
            vb3prev = vb3
        end
        stepsize /= 2
    end
    noisy, truth = generate(c)
    output1 = Convolve!(noisy[1:c, :, :, :], output1, conv1)
    output1 = ReLU!(output1)
    output2 = Convolve!(output1, output2, conv2)
    output2 = ReLU!(output2)
    output3 = Convolve!(output2, output3, conv3)

    output3 = max.(0, output3)
    output3 = min.(1, output3)

    for i = axes(noisy, 1)
        save("Original$i.png", colorview(RGB, noisy[i, :, :, :]))
        save("Output$i.png", colorview(RGB, output3[i, :, :, :]))
        save("Truth$i.png", colorview(RGB, truth[i, :, :, :, :]))
    end
    return noisy, output3, truth
end

function downscale(x)
    y = zeros(Float32, (size(x, 1), size(x, 2), size(x, 3) ÷ 2, size(x, 4)÷2))
    for i = CartesianIndices(y)
        h = i[3] * 2 - 1
        w = i[4] * 2 - 1
        y[i] += x[i[1], i[2], h, w]
        y[i] += x[i[1], i[2], h+1, w]
        y[i] += x[i[1], i[2], h, w+1]
        y[i] += x[i[1], i[2], h+1, w+1]
    end
    y .*= 0.25
    return y
end

function simplePoolingTest()
    f = 8
    c = 16
    conv1 = Convolution(3, 3, 3, f)
    conv2 = Convolution(3, 3, f, 3)
    step1 = Convolution(3, 3, 3, f)
    step2 = Convolution(3, 3, f, 3)
    adam1 = AdamData(conv1)
    adam2 = AdamData(conv2)

    noisy, truth = generate(50000)
    #noisy = truth
    truth = downscale(truth)

    o1a = zeros(Float32, (c, f, 32, 32))
    o1b = zeros(Float32, (c, f, 16, 16))
    o2 = zeros(Float32, (c, 3, 16, 16))
    d1 = zeros(Float32, (c, f, 32, 32))
    d1t = zeros(Float32, (c, f, 32, 32))
    delta1 = zeros(Float32, (c, f, 32, 32))
    delta2 = zeros(Float32, (c, 3, 16, 16))
    for j = 1:10
        for i = 1:c:50000
            o1a = Convolve!(noisy[i:i+c-1, :,:,:], o1a, conv1)
            d1 = dMaxPool!(d1, o1a) .* dReLU!(d1t, o1a)
            o1b = MaxPool!(ReLU!(o1b), o1a)
            o2 = Convolve!(o1b, o2, conv2)

            delta2 = o2 - truth[i:i+c-1, :,:,:]

            println(sum(abs.(delta2)) / (c * 3 * 16 * 16))

            delta1 = MaxPoolingBackConvolve!(delta2, delta1, d1, conv2)

            step1 = WeightStep!(noisy[i:i+c-1, :,:,:], delta1, step1)
            step2 = WeightStep!(o1b, delta2, step2)

            adam1 = AdamUpdate!(step1, adam1)
            adam2 = AdamUpdate!(step2, adam2)

            conv1 = WeightUpdate!(conv1, adam1)
            conv2 = WeightUpdate!(conv2, adam2)
        end
    end
end

function simplePoolingAndUpscaleTest()
    f = 12
    c = 16
    conv1 = Convolution(3, 3, 3, f)
    conv2 = Convolution(3, 3, f, 12)
    conv3 = Convolution(3, 3, 3, 3)
    step1 = Convolution(3, 3, 3, f)
    step2 = Convolution(3, 3, f, 12)
    adam1 = AdamData(conv1)
    adam2 = AdamData(conv2)

    noisy, truth = generate(50000)
    #noisy = truth

    o1a = zeros(Float32, (c, f, 32, 32))
    o1b = zeros(Float32, (c, f, 16, 16))
    o2 = zeros(Float32, (c, 12, 16, 16))
    o2d2s = zeros(Float32, (c, 3, 32, 32))
    o3 = zeros(Float32, (c, 3, 32, 32))
    d1 = zeros(Float32, (c, f, 32, 32))
    d1t = zeros(Float32, (c, f, 32, 32))
    d2 = zeros(Float32, (c, f, 32, 32))
    delta1 = zeros(Float32, (c, f, 32, 32))
    delta2 = zeros(Float32, (c, 12, 16, 16))
    delta2d2s = zeros(Float32, (c, 3, 32, 32))
    delta3 = zeros(Float32, (c, 3, 32, 32))
    for j = 1:10
        for i = 1:c:50000
            o1a = Convolve!(noisy[i:i+c-1, :,:,:], o1a, conv1)
            d1 = dMaxPool!(d1, o1a) .* dReLU!(d1t, o1a)
            o1b = MaxPool!(ReLU!(o1b), o1a)
            o2 = Convolve!(o1b, o2, conv2)

            o2d2s = DepthToSpace!(o2, o2d2s)

            delta2d2s = o2d2s - truth[i:i+c-1, :,:,:]
            delta2 = SpaceToDepth!(delta2d2s, delta2)

            println(sum(abs.(delta2)) / (c * 3 * 32 * 32))

            delta1 = MaxPoolingBackConvolve!(delta2, delta1, d1, conv2)

            step1 = WeightStep!(noisy[i:i+c-1, :,:,:], delta1, step1)
            step2 = WeightStep!(o1b, delta2, step2)

            adam1 = AdamUpdate!(step1, adam1)
            adam2 = AdamUpdate!(step2, adam2)

            conv1 = WeightUpdate!(conv1, adam1)
            conv2 = WeightUpdate!(conv2, adam2)
        end
    end
end

simpleNeuralNetTest()
#simplePoolingAndUpscaleTest()