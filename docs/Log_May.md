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

17/05/25
With numpy implementation
Train Data Eval:
  Num examples: 2057  Total loss: 189.761195765  Mean loss @ 1: 0.092251432
Validate Data Eval:
  Num examples: 2071  Total loss: 205.673016185  Mean loss @ 1: 0.099310969
time:0.49419689178466797 order:2

Train Data Eval:
  Num examples: 2057  Total loss: 292.948385792  Mean loss @ 1: 0.142415355
Validate Data Eval:
  Num examples: 2071  Total loss: 313.804377516  Mean loss @ 1: 0.151523118
time:0.8582067489624023 order:3

Train Data Eval:
  Num examples: 2057  Total loss: 1684.335626368  Mean loss @ 1: 0.818831126
Validate Data Eval:
  Num examples: 2071  Total loss: 1789.041629169  Mean loss @ 1: 0.863853998
time:1.5161230564117432 order:4

Train Data Eval:
  Num examples: 2057  Total loss: 1183.720373888  Mean loss @ 1: 0.575459589
Validate Data Eval:
  Num examples: 2071  Total loss: 1266.313067029  Mean loss @ 1: 0.611450057
time:2.437915086746216 order:5

Train Data Eval:
  Num examples: 2057  Total loss: 2152.610557513  Mean loss @ 1: 1.046480582
Validate Data Eval:
  Num examples: 2071  Total loss: 2286.076853951  Mean loss @ 1: 1.103851692
time:3.606255054473877 order:6

Train Data Eval:
  Num examples: 2057  Total loss: 574.878534730  Mean loss @ 1: 0.279474251
Validate Data Eval:
  Num examples: 2071  Total loss: 627.113708744  Mean loss @ 1: 0.302807199
time:5.494869709014893 order:7

Train Data Eval:
  Num examples: 2057  Total loss: 244.016066616  Mean loss @ 1: 0.118627159
Validate Data Eval:
  Num examples: 2071  Total loss: 259.774471547  Mean loss @ 1: 0.125434318
time:8.096030712127686 order:8

Train Data Eval:
  Num examples: 2057  Total loss: 577.477422031  Mean loss @ 1: 0.280737687
Validate Data Eval:
  Num examples: 2071  Total loss: 612.367111990  Mean loss @ 1: 0.295686679
time:11.46686315536499 order:9



With default implementation
Train Data Eval:
  Num examples: 2057  Total loss: 189.761195765  Mean loss @ 1: 0.092251432
Validate Data Eval:
  Num examples: 2071  Total loss: 205.673016185  Mean loss @ 1: 0.099310969
time:0.5423338413238525 order:2

Train Data Eval:
  Num examples: 2057  Total loss: 292.948385792  Mean loss @ 1: 0.142415355
Validate Data Eval:
  Num examples: 2071  Total loss: 313.804377516  Mean loss @ 1: 0.151523118
time:0.8829007148742676 order:3

Train Data Eval:
  Num examples: 2057  Total loss: 1684.335626368  Mean loss @ 1: 0.818831126
Validate Data Eval:
  Num examples: 2071  Total loss: 1789.041629169  Mean loss @ 1: 0.863853998
time:1.5261614322662354 order:4

Train Data Eval:
  Num examples: 2057  Total loss: 1183.720373888  Mean loss @ 1: 0.575459589
Validate Data Eval:
  Num examples: 2071  Total loss: 1266.313067029  Mean loss @ 1: 0.611450057
time:2.6180903911590576 order:5

Train Data Eval:
  Num examples: 2057  Total loss: 2152.610557513  Mean loss @ 1: 1.046480582
Validate Data Eval:
  Num examples: 2071  Total loss: 2286.076853951  Mean loss @ 1: 1.103851692
time:3.8972697257995605 order:6

Train Data Eval:
  Num examples: 2057  Total loss: 574.878534730  Mean loss @ 1: 0.279474251
Validate Data Eval:
  Num examples: 2071  Total loss: 627.113708744  Mean loss @ 1: 0.302807199
time:5.884138107299805 order:7

Train Data Eval:
  Num examples: 2057  Total loss: 244.016066616  Mean loss @ 1: 0.118627159
Validate Data Eval:
  Num examples: 2071  Total loss: 259.774471547  Mean loss @ 1: 0.125434318
time:12.904129981994629 order:8

Train Data Eval:
  Num examples: 2057  Total loss: 577.477422031  Mean loss @ 1: 0.280737687
Validate Data Eval:
  Num examples: 2071  Total loss: 612.367111990  Mean loss @ 1: 0.295686679
time:15.243883609771729 order:9