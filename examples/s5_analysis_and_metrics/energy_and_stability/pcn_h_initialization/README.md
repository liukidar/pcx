Results were achieved by running hyperparameters tuning using `stune`.

The found hyperparameters are as following:
- Forward initialization:
    - T=3: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.017645706311565725, hp/optim/x/momentum: 0.05, hp/optim/w/lr: 0.00029836169585628326, hp/optim/w/wd: 0.001307773142925482]`
    - T=5: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.010707247994012799, hp/optim/x/momentum: 0.15000000000000002, hp/optim/w/lr: 0.000250810531013389, hp/optim/w/wd: 0.00040768138013852647]`
    - T=7: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.011634370009349783, hp/optim/x/momentum: 0.2, hp/optim/w/lr: 0.0002551975165828075, hp/optim/w/wd: 1.2481848298314534e-05]`
    - T=9: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.012353149258473414, hp/optim/x/momentum: 0.0, hp/optim/w/lr: 0.00028013598038820556, hp/optim/w/wd: 0.00017790347367183153]`
    - T=11: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.010073678676718305, hp/optim/x/momentum: 0.0, hp/optim/w/lr: 0.000299878444821947, hp/optim/w/wd: 2.0002691625448918e-05]`
    - T=15: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.010049238099531978, hp/optim/x/momentum: 0.05, hp/optim/w/lr: 0.00026875770087137174, hp/optim/w/wd: 0.0006046395352801334]`
    - T=20: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.01039164117033675, hp/optim/x/momentum: 0.2, hp/optim/w/lr: 0.00023411900438700673, hp/optim/w/wd: 0.009900931761050152]`
- Zero initialization:
    - T=3: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.11036598928934052, hp/optim/x/momentum: 0.45, hp/optim/w/lr: 0.0001720941249469424, hp/optim/w/wd: 2.3523077868141024e-05]`
    - T=5: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.1249345615856544, hp/optim/x/momentum: 0.75, hp/optim/w/lr: 5.600901289523681e-05, hp/optim/w/wd: 0.0006430435629443783]`
    - T=7: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.31073058432495276, hp/optim/x/momentum: 0.25, hp/optim/w/lr: 8.642938914254075e-05, hp/optim/w/wd: 0.0009372314002807972]`
    - T=9: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.35941863117294104, hp/optim/x/momentum: 0.15000000000000002, hp/optim/w/lr: 6.677152825441371e-05, hp/optim/w/wd: 1.2552428150513801e-05]`
    - T=11: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.27889021424657806, hp/optim/x/momentum: 0.4, hp/optim/w/lr: 4.227643740455322e-05, hp/optim/w/wd: 0.00038398065912067657]`
    - T=15: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.2792325235869229, hp/optim/x/momentum: 0.35000000000000003, hp/optim/w/lr: 0.0001779663196416802, hp/optim/w/wd: 1.6567812264230678e-05]`
    - T=20: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.21232253118339392, hp/optim/x/momentum: 0.45, hp/optim/w/lr: 3.891464395163487e-05, hp/optim/w/wd: 2.620471473848808e-05]`
- Normal initialization:
    - T=3: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.3149283941414952, hp/optim/x/momentum: 0.5, hp/optim/w/lr: 5.603921714076118e-05, hp/optim/w/wd: 0.0003336486000090618]`
    - T=5: `Params = [hp/act_fn: gelu, hp/optim/x/lr: 0.15471924981830182, hp/optim/x/momentum: 0.6000000000000001, hp/optim/w/lr: 0.00016357019375209944, hp/optim/w/wd: 0.0031812868127717303]`
    - T=7: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.2379512263064572, hp/optim/x/momentum: 0.35000000000000003, hp/optim/w/lr: 4.799772009342855e-05, hp/optim/w/wd: 0.00034930728274418417]`
    - T=9: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.31805140851879976, hp/optim/x/momentum: 0.25, hp/optim/w/lr: 0.0001736249511939323, hp/optim/w/wd: 0.005844991729379492]`
    - T=11: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.30318057442118573, hp/optim/x/momentum: 0.35000000000000003, hp/optim/w/lr: 6.82385010154935e-05, hp/optim/w/wd: 0.004611892041495239]`
    - T=15: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.28531857882110945, hp/optim/x/momentum: 0.4, hp/optim/w/lr: 0.00018988454516116981, hp/optim/w/wd: 1.9379565752841875e-05]`
    - T=20: `Params = [hp/act_fn: leaky_relu, hp/optim/x/lr: 0.34784140273082864, hp/optim/x/momentum: 0.2, hp/optim/w/lr: 3.780568458017943e-05, hp/optim/w/wd: 0.0005471851349459166]`

To run each specific experiment, simply substitute the `default` in `EP.yaml` and run the desired python script.