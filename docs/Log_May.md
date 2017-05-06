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
