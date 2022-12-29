include("DataAndGenerate.jl")
include("Convolution.jl")
include("Activations.jl")
include("AdamUpdate.jl")
include("PixelShuffle.jl")

#This function still seems to be incorrect, it was meant to be a sanity check, but instead proved to be the opposite
function GradientCheck()
    #http://deeplearning.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
    #https://datascience-enthusiast.com/DL/Improving_DeepNeural_Networks_Gradient_Checking.html
    conv1 = Convolution(3, 3, 3, 3)
    conv2 = Convolution(3, 3, 3, 3)


    # Compute conv1 and conv2 with a small perturbation in the weights
    eps = Float32(1e-7)
    conv1_l = Convolution(conv1.weights .- eps, conv1.bias .- eps)
    conv2_l = Convolution(conv2.weights .- eps, conv2.bias .- eps)
    conv1_r = Convolution(conv1.weights .+ eps, conv1.bias .+ eps)
    conv2_r = Convolution(conv2.weights .+ eps, conv2.bias .+ eps)

    # Generate input data
    noisy, truth = generate(16)

    # Compute forward pass with original convolutional layers
    o1 = zeros(Float32, (16, 3, 32, 32))
    o2 = zeros(Float32, (16, 3, 32, 32))
    delta1 = zeros(Float32, (16, 3, 32, 32))
    delta2 = zeros(Float32, (16, 3, 32, 32))
    d1 = ones(Float32, (16, 3, 32, 32))
    
    o1 = Convolve!(noisy, o1, conv1)
    o2 = Convolve!(o1, o2, conv2)

    # Compute forward pass with perturbed convolutional layers
    o1_l = zeros(Float32, (16, 3, 32, 32))
    o1_r = zeros(Float32, (16, 3, 32, 32))
    o2_l = zeros(Float32, (16, 3, 32, 32))
    o2_r = zeros(Float32, (16, 3, 32, 32))

    """o1_l = Convolve!(noisy, o1_l, conv1)
    o1_r = Convolve!(noisy, o1_r, conv1)
    o2_l = Convolve!(o1_l  .- eps, o2_l, conv2)
    o2_r = Convolve!(o1_r .+ eps, o2_r, conv2)"""

    o1_l = Convolve!(noisy, o1_l, conv1)
    o1_r = Convolve!(noisy, o1_r, conv1)
    o2_l = Convolve!(o1_l, o2_l, conv2_l)
    o2_r = Convolve!(o1_r, o2_r, conv2_r)

    # Compute gradient of loss with respect to output of second convolutional layer
    delta1 = BackConvolve!(2 .*(o2 - truth), delta1, d1, conv2)

    # Compute numerical gradient with perturbed forward pass
    loss_l = ((o2_l - truth) .* (o2_l - truth))
    loss_r = ((o2_r - truth) .* (o2_r - truth))
    grad = ((loss_r - loss_l) ./ (2 * eps)) .* ones(size(delta1))

    # Compare gradient to computed gradient
    numerator = sqrt(sum((abs.(delta1).-abs.(grad)) .* (abs.(delta1).-abs.(grad))))
    denominator = sqrt(sum(delta1 .* delta1)) + sqrt(sum(grad .* grad))
    #numerator = sum(delta1.*grad)
    #denominator = sqrt(sum(delta1.*delta1)) * sqrt(sum(grad.*grad))
    println(numerator / denominator)
end

GradientCheck()