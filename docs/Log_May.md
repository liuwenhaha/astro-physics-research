5/3/2017
Make the exposure plot

5/4/2017
tensorflow interpolation:
    - pixel-wise
    - psf-wise
Step 1:
    use 1-layer feed-forward neural network
    with hidden unit given by:
    $n_h = 2 \sqrt(n_i + n_o) + \alpha$

5/6/2017
program:
    cache psf_data into file
    implement psfwise interpolation
        first implement the exposure interpolation
Idea:
    predict:
        write a function to generate PSF predictions from TF
        maybe write inside the tf__interpolation file
    evaluate: write a evaluate method in PSF class
        input: list of method name
        output: pixel-wise MSE over validate set
    number of density:
        a way to evaluate interpolation over number of density is to use the train to validate set
        first use train_data_ratio = 0.5
        rename sub-directory under cache to include train_data_ratio later

5/10/2017
TODO:
    implement linear interpolation
        calculate the mse
    check if better loss can be found(divide by original psf)
    visualize hidden1 hidden2
    hidden1 36 hidden2 144
    use trained model to make predictions

Results:
    learning rate
        0.1/1/10 seems that bigger the better
    poly-1 result
        Train Data Eval:
          Num examples: 2057  Total loss: 0.000047646  Mean loss @ 1: 0.000000023
        Validate Data Eval:
          Num examples: 2071  Total loss: 0.000046190  Mean loss @ 1: 0.000000022
        Train Data Eval:
          Num examples: 2057  Total loss: 225.809673050  Mean loss @ 1: 0.109776214
        Validate Data Eval:
          Num examples: 2071  Total loss: 220.401262499  Mean loss @ 1: 0.106422628
