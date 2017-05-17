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
    l2_lr1_ms4000_h1.36_h2.144_bs100:
        Training Data Eval:
          Num examples: 2000  Total loss: 0.000967675  Mean loss @ 1: 0.000000484
        Validation Data Eval:
          Num examples: 2000  Total loss: 0.000947204  Mean loss @ 1: 0.000000474
    l2_lr0.1_ms4000_h1.36_h2.144_bs100:
        Training Data Eval:
          Num examples: 2000  Total loss: 0.001110022  Mean loss @ 1: 0.000000555
        Validation Data Eval:
          Num examples: 2000  Total loss: 0.001062898  Mean loss @ 1: 0.000000531
    l2_lr10_ms4000_h1.36_h2.144_bs100:
        Training Data Eval:
          Num examples: 2000  Total loss: 0.086969081  Mean loss @ 1: 0.000043485
        Validation Data Eval:
          Num examples: 2000  Total loss: 0.086431238  Mean loss @ 1: 0.000043216

need more iteration:
    2-layer lr10

17/05/17
Train Data Eval:
  Num examples: 2057  Total loss: 3208995957053830144.000000000  Mean loss @ 1: 1560036926132148.750000000
Validate Data Eval:
  Num examples: 2071  Total loss: 13017802912125069312.000000000  Mean loss @ 1: 6285757079732047.000000000
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_2/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 8331506684975491579904.000000000  Mean loss @ 1: 4050319244032810496.000000000
Validate Data Eval:
  Num examples: 2071  Total loss: 33798083247930230177792.000000000  Mean loss @ 1: 16319692538836422656.000000000
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_3/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 390805275544182.250000000  Mean loss @ 1: 189987980332.611694336
Validate Data Eval:
  Num examples: 2071  Total loss: 1585360763844029.000000000  Mean loss @ 1: 765504955984.562500000
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_4/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 179938119323.596893311  Mean loss @ 1: 87475993.837431639
Validate Data Eval:
  Num examples: 2071  Total loss: 729820331159.741210938  Mean loss @ 1: 352399966.759894371
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_5/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178172264359.908325195  Mean loss @ 1: 86617532.503601521
Validate Data Eval:
  Num examples: 2071  Total loss: 722655784305.106689453  Mean loss @ 1: 348940504.251620829
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_6/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171926294.555297852  Mean loss @ 1: 86617368.154864028
Validate Data Eval:
  Num examples: 2071  Total loss: 722654612053.152221680  Mean loss @ 1: 348939938.219774127
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_7/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928304.840240479  Mean loss @ 1: 86617369.132153735
Validate Data Eval:
  Num examples: 2071  Total loss: 722654726495.454101562  Mean loss @ 1: 348939993.479214907
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_8/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928189.863677979  Mean loss @ 1: 86617369.076258466
Validate Data Eval:
  Num examples: 2071  Total loss: 722654806689.169433594  Mean loss @ 1: 348940032.201433837
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_9/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928664.412994385  Mean loss @ 1: 86617369.306958184
Validate Data Eval:
  Num examples: 2071  Total loss: 722654865807.970581055  Mean loss @ 1: 348940060.747450769
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_10/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928485.194396973  Mean loss @ 1: 86617369.219831988
Validate Data Eval:
  Num examples: 2071  Total loss: 722654981309.435302734  Mean loss @ 1: 348940116.518317401
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_11/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928498.813293457  Mean loss @ 1: 86617369.226452738
Validate Data Eval:
  Num examples: 2071  Total loss: 722655004032.199584961  Mean loss @ 1: 348940127.490197778
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_12/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928107.800781250  Mean loss @ 1: 86617369.036364019
Validate Data Eval:
  Num examples: 2071  Total loss: 722655008132.525634766  Mean loss @ 1: 348940129.470075130
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_13/predictions.fits
Train Data Eval:
  Num examples: 2057  Total loss: 178171928754.484039307  Mean loss @ 1: 86617369.350745767
Validate Data Eval:
  Num examples: 2071  Total loss: 722655002471.663452148  Mean loss @ 1: 348940126.736679614
poly1 predictions saved to assets/predictions/w2m0m0_831555/poly_14/predictions.fits