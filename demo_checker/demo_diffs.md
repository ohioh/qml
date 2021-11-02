Last update: 2021-11-02  17:29:50 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_vqe.html](#demo0)
2. [tutorial_rosalin.html](#demo1)
3. [tutorial_gaussian_transformation.html](#demo2)
4. [tutorial_rotoselect.html](#demo3)
5. [tutorial_gbs.html](#demo4)
6. [tutorial_noisy_circuits.html](#demo5)
7. [tutorial_jax_transformations.html](#demo6)
8. [tutorial_qgrnn.html](#demo7)
9. [tutorial_pasqal.html](#demo8)
10. [tutorial_quanvolution.html](#demo9)
11. [tutorial_ensemble_multi_qpu.html](#demo10)
12. [tutorial_multiclass_classification.html](#demo11)
13. [tutorial_adaptive_circuits.html](#demo12)
14. [tutorial_data_reuploading_classifier.html](#demo13)
15. [tutorial_classical_shadows.html](#demo14)
16. [tutorial_vqe_spin_sectors.html](#demo15)
17. [tutorial_vqe_parallel.html](#demo16)
18. [tutorial_qnn_module_tf.html](#demo17)
19. [tutorial_backprop.html](#demo18)
20. [tutorial_state_preparation.html](#demo19)
21. [tutorial_general_parshift.html](#demo20)
22. [tutorial_doubly_stochastic.html](#demo21)
23. [tutorial_variational_classifier.html](#demo22)
24. [tutorial_falqon.html](#demo23)
25. [tutorial_qaoa_intro.html](#demo24)
26. [tutorial_vqe_qng.html](#demo25)
27. [tutorial_quantum_natural_gradient.html](#demo26)
28. [tutorial_quantum_transfer_learning.html](#demo27)
29. [tutorial_chemical_reactions.html](#demo28)
30. [tutorial_mol_geo_opt.html](#demo29)
31. [tutorial_QGAN.html](#demo30)
32. [tutorial_measurement_optimize.html](#demo31)
33. [tutorial_expressivity_fourier_series.html](#demo32)
34. [tutorial_vqt.html](#demo33)
35. [tutorial_quantum_chemistry.html](#demo34)
36. [tutorial_local_cost_functions.html](#demo35)


Number of demos different/all demos: 36/54

## 1. tutorial_vqe.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe.html):

```
Step = 0,  Energy = -1.12799983 Ha
Step = 2,  Energy = -1.13466246 Ha
Step = 4,  Energy = -1.13590595 Ha
Step = 6,  Energy = -1.13613667 Ha
Step = 8,  Energy = -1.13617944 Ha
Step = 10,  Energy = -1.13618736 Ha
Step = 12,  Energy = -1.13618883 Ha
Final value of the ground-state energy = -1.13618883 Ha
Optimal value of the circuit parameter = 0.2089
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step = 0,  Energy = -1.12799983 Ha
Step = 2,  Energy = -1.13466246 Ha
Step = 4,  Energy = -1.13590595 Ha
Step = 6,  Energy = -1.13613667 Ha
Step = 8,  Energy = -1.13617944 Ha
Step = 10,  Energy = -1.13618736 Ha
Step = 12,  Energy = -1.13618883 Ha
Final value of the ground-state energy = -1.13618883 Ha
```

---

## 2. tutorial_rosalin.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost = -0.47971271815214855 shots used = 0
Step 1: cost = -1.6879973520840041 shots used = 8000
Step 2: cost = -2.437928256197112 shots used = 16000
Step 3: cost = -2.9300968884147647 shots used = 24000
Step 4: cost = -3.7779069617997116 shots used = 32000
Step 5: cost = -3.8889841568955115 shots used = 40000
Step 6: cost = -4.508059711766957 shots used = 48000
Step 7: cost = -4.71114219758592 shots used = 56000
Step 8: cost = -4.984457128293103 shots used = 64000
Step 9: cost = -5.597084424095087 shots used = 72000
Step 10: cost = -5.456976403687039 shots used = 80000
Step 11: cost = -5.736752824027413 shots used = 88000
Step 12: cost = -6.220317925041974 shots used = 96000
Step 13: cost = -6.45162161927903 shots used = 104000
Step 14: cost = -6.563539211112225 shots used = 112000
Step 15: cost = -6.487339064303318 shots used = 120000
Step 16: cost = -6.69261841162329 shots used = 128000
Step 17: cost = -6.909230576241427 shots used = 136000
Step 18: cost = -7.05156660241221 shots used = 144000
Step 19: cost = -7.163688069859358 shots used = 152000
Step 20: cost = -7.191791478058647 shots used = 160000
Step 21: cost = -7.191694602776715 shots used = 168000
Step 22: cost = -7.430122007574104 shots used = 176000
Step 23: cost = -7.245621601209081 shots used = 184000
Step 24: cost = -7.539044265851978 shots used = 192000
Step 25: cost = -7.532847998808006 shots used = 200000
Step 26: cost = -7.44257222073886 shots used = 208000
Step 27: cost = -7.439951968648378 shots used = 216000
Step 28: cost = -7.734568855081575 shots used = 224000
Step 29: cost = -7.618221322585628 shots used = 232000
Step 30: cost = -7.651544920606065 shots used = 240000
Step 31: cost = -7.5069088885777155 shots used = 248000
Step 32: cost = -7.780301321189146 shots used = 256000
Step 33: cost = -7.4456447455856445 shots used = 264000
Step 34: cost = -7.403560444278863 shots used = 272000
Step 35: cost = -7.666718876831026 shots used = 280000
Step 36: cost = -7.7178910518866415 shots used = 288000
Step 37: cost = -7.375680885292107 shots used = 296000
Step 38: cost = -7.665568049279896 shots used = 304000
Step 39: cost = -7.568101693343673 shots used = 312000
Step 40: cost = -7.524188200359864 shots used = 320000
Step 41: cost = -7.525528734255245 shots used = 328000
Step 42: cost = -7.57734861403185 shots used = 336000
Step 43: cost = -7.76844833198197 shots used = 344000
Step 44: cost = -7.797619087079373 shots used = 352000
Step 45: cost = -7.879148884805528 shots used = 360000
Step 46: cost = -7.744030492750696 shots used = 368000
Step 47: cost = -7.6484739221198765 shots used = 376000
Step 48: cost = -7.679623095926702 shots used = 384000
Step 49: cost = -7.607476988501242 shots used = 392000
Step 50: cost = -7.856041856821188 shots used = 400000
Step 51: cost = -7.644473030321983 shots used = 408000
Step 52: cost = -7.593159311741706 shots used = 416000
Step 53: cost = -7.606939212888227 shots used = 424000
Step 54: cost = -7.621128949485829 shots used = 432000
Step 55: cost = -7.743568287057952 shots used = 440000
Step 56: cost = -7.6325929460598525 shots used = 448000
Step 57: cost = -7.718256562367575 shots used = 456000
Step 58: cost = -7.861601938446393 shots used = 464000
Step 59: cost = -7.666115854972354 shots used = 472000
Step 60: cost = -7.644148944168839 shots used = 480000
Step 61: cost = -7.771569192260795 shots used = 488000
Step 62: cost = -7.776898446282362 shots used = 496000
Step 63: cost = -7.711006891533269 shots used = 504000
Step 64: cost = -7.748650044666392 shots used = 512000
Step 65: cost = -7.690723991927554 shots used = 520000
Step 66: cost = -7.694117031088106 shots used = 528000
Step 67: cost = -7.793250125674997 shots used = 536000
Step 68: cost = -7.926049735334674 shots used = 544000
Step 69: cost = -7.686292326080605 shots used = 552000
Step 70: cost = -7.745774212716911 shots used = 560000
Step 71: cost = -7.625346751584894 shots used = 568000
Step 72: cost = -7.846664469958039 shots used = 576000
Step 73: cost = -7.860275655123486 shots used = 584000
Step 74: cost = -7.593043619614097 shots used = 592000
Step 75: cost = -7.7969799318129045 shots used = 600000
Step 76: cost = -7.837545360539077 shots used = 608000
Step 77: cost = -7.845253964960701 shots used = 616000
Step 78: cost = -7.941652692590529 shots used = 624000
Step 79: cost = -7.967099906804574 shots used = 632000
Step 80: cost = -7.803163356121793 shots used = 640000
Step 81: cost = -7.665600401510319 shots used = 648000
Step 82: cost = -8.09158124610039 shots used = 656000
Step 83: cost = -7.774883584668083 shots used = 664000
Step 84: cost = -7.758175214036924 shots used = 672000
Step 85: cost = -7.9169924228411865 shots used = 680000
Step 86: cost = -7.670199051467696 shots used = 688000
Step 87: cost = -8.085682024006845 shots used = 696000
Step 88: cost = -7.8433919424579095 shots used = 704000
Step 89: cost = -7.755236580472145 shots used = 712000
Step 90: cost = -7.847624689390126 shots used = 720000
Step 91: cost = -8.122239105086607 shots used = 728000
Step 92: cost = -7.922374192271718 shots used = 736000
Step 93: cost = -7.904676929818973 shots used = 744000
Step 94: cost = -7.909417248833883 shots used = 752000
Step 95: cost = -8.06033491620787 shots used = 760000
Step 96: cost = -7.765636196903123 shots used = 768000
Step 97: cost = -7.801666008865329 shots used = 776000
Step 98: cost = -8.066513329432457 shots used = 784000
Step 99: cost = -7.8942080196569675 shots used = 792000
Step 0: cost = -0.38250000000000006 shots used = 0
Step 1: cost = -1.7450000000000006 shots used = 8000
Step 2: cost = -2.54875 shots used = 16000
Step 3: cost = -2.91 shots used = 24000
Step 4: cost = -3.4762500000000003 shots used = 32000
Step 5: cost = -4.08875 shots used = 40000
Step 6: cost = -4.586250000000001 shots used = 48000
Step 7: cost = -4.805 shots used = 56000
Step 8: cost = -4.925 shots used = 64000
Step 9: cost = -5.385000000000001 shots used = 72000
Step 10: cost = -5.4725 shots used = 80000
Step 11: cost = -5.63875 shots used = 88000
Step 12: cost = -5.796250000000001 shots used = 96000
Step 13: cost = -6.308750000000001 shots used = 104000
Step 14: cost = -6.2524999999999995 shots used = 112000
Step 15: cost = -6.706249999999999 shots used = 120000
Step 16: cost = -6.711250000000001 shots used = 128000
Step 17: cost = -6.803749999999999 shots used = 136000
Step 18: cost = -6.94375 shots used = 144000
Step 19: cost = -7.2837499999999995 shots used = 152000
Step 20: cost = -7.4 shots used = 160000
Step 21: cost = -7.38375 shots used = 168000
Step 22: cost = -7.40125 shots used = 176000
Step 23: cost = -7.4775 shots used = 184000
Step 24: cost = -7.58 shots used = 192000
Step 25: cost = -7.623749999999999 shots used = 200000
Step 26: cost = -7.49625 shots used = 208000
Step 27: cost = -7.58375 shots used = 216000
Step 28: cost = -7.6312500000000005 shots used = 224000
Step 29: cost = -7.13375 shots used = 232000
Step 30: cost = -7.47 shots used = 240000
Step 31: cost = -7.6075 shots used = 248000
Step 32: cost = -7.34875 shots used = 256000
Step 33: cost = -7.6525 shots used = 264000
Step 34: cost = -7.572500000000001 shots used = 272000
Step 35: cost = -7.390000000000001 shots used = 280000
Step 36: cost = -7.76375 shots used = 288000
Step 37: cost = -7.49 shots used = 296000
Step 38: cost = -7.61625 shots used = 304000
Step 39: cost = -7.695 shots used = 312000
Step 40: cost = -7.702499999999999 shots used = 320000
Step 41: cost = -7.59625 shots used = 328000
Step 42: cost = -7.733750000000001 shots used = 336000
Step 43: cost = -7.6875 shots used = 344000
Step 44: cost = -7.75875 shots used = 352000
Step 45: cost = -7.796250000000001 shots used = 360000
Step 46: cost = -7.7387500000000005 shots used = 368000
Step 47: cost = -7.92375 shots used = 376000
Step 48: cost = -7.6225 shots used = 384000
Step 49: cost = -7.8425 shots used = 392000
Step 50: cost = -7.74 shots used = 400000
Step 51: cost = -7.661250000000001 shots used = 408000
Step 52: cost = -7.786250000000001 shots used = 416000
Step 53: cost = -7.78875 shots used = 424000
Step 54: cost = -7.62375 shots used = 432000
Step 55: cost = -7.9375 shots used = 440000
Step 56: cost = -7.71625 shots used = 448000
Step 57: cost = -7.72375 shots used = 456000
Step 58: cost = -7.741250000000001 shots used = 464000
Step 59: cost = -7.811249999999999 shots used = 472000
Step 60: cost = -7.89 shots used = 480000
Step 61: cost = -7.74 shots used = 488000
Step 62: cost = -7.751250000000001 shots used = 496000
Step 63: cost = -7.71875 shots used = 504000
Step 64: cost = -7.695 shots used = 512000
Step 65: cost = -7.7325 shots used = 520000
Step 66: cost = -7.819999999999999 shots used = 528000
Step 67: cost = -7.981249999999999 shots used = 536000
Step 68: cost = -7.8 shots used = 544000
Step 69: cost = -7.89 shots used = 552000
Step 70: cost = -7.7125 shots used = 560000
Step 71: cost = -7.993750000000001 shots used = 568000
Step 72: cost = -7.772499999999999 shots used = 576000
Step 73: cost = -8.01125 shots used = 584000
Step 74: cost = -8.116249999999999 shots used = 592000
Step 75: cost = -7.9662500000000005 shots used = 600000
Step 76: cost = -7.7125 shots used = 608000
Step 77: cost = -7.8925 shots used = 616000
Step 78: cost = -7.967499999999999 shots used = 624000
Step 79: cost = -7.91375 shots used = 632000
Step 80: cost = -7.797499999999999 shots used = 640000
Step 81: cost = -7.9975000000000005 shots used = 648000
Step 82: cost = -7.99 shots used = 656000
Step 83: cost = -7.7124999999999995 shots used = 664000
Step 84: cost = -7.76875 shots used = 672000
Step 85: cost = -7.62 shots used = 680000
Step 86: cost = -7.822500000000001 shots used = 688000
Step 87: cost = -7.74625 shots used = 696000
Step 88: cost = -7.9137499999999985 shots used = 704000
Step 89: cost = -7.86125 shots used = 712000
Step 90: cost = -7.975 shots used = 720000
Step 91: cost = -7.89375 shots used = 728000
Step 92: cost = -8.1075 shots used = 736000
Step 93: cost = -7.775 shots used = 744000
Step 94: cost = -7.8999999999999995 shots used = 752000
Step 95: cost = -7.85625 shots used = 760000
Step 96: cost = -7.925000000000001 shots used = 768000
Step 97: cost = -8.0 shots used = 776000
Step 98: cost = -7.825000000000001 shots used = 784000
Step 99: cost = -7.999999999999999 shots used = 792000
Step 0: cost = -5.976611864639143, shots_used = 240
Step 1: cost = -3.9696542358660727, shots_used = 288
Step 2: cost = -4.960189727105254, shots_used = 360
Step 3: cost = -4.580003760087767, shots_used = 456
Step 4: cost = -2.230216749128693, shots_used = 552
Step 5: cost = -3.6390262209635624, shots_used = 696
Step 6: cost = -6.407579837465835, shots_used = 1050
Step 7: cost = -7.4366536874312565, shots_used = 1578
Step 8: cost = -7.259604321778904, shots_used = 2250
Step 9: cost = -7.062132684694287, shots_used = 2970
Step 10: cost = -7.5539381823528915, shots_used = 3738
Step 11: cost = -7.530120251217975, shots_used = 4866
Step 12: cost = -7.620064018172076, shots_used = 6474
Step 13: cost = -7.749105026853709, shots_used = 8288
Step 14: cost = -7.7584669100105454, shots_used = 10388
Step 15: cost = -7.547668090788587, shots_used = 12404
Step 16: cost = -7.802606000681813, shots_used = 14660
Step 17: cost = -7.819375105495885, shots_used = 17180
Step 18: cost = -7.813893056373781, shots_used = 19700
Step 19: cost = -7.818976697763795, shots_used = 22796
Step 20: cost = -7.847655565015213, shots_used = 26372
Step 21: cost = -7.854512274045721, shots_used = 30810
Step 22: cost = -7.855665819254089, shots_used = 35538
Step 23: cost = -7.843276666680198, shots_used = 40770
Step 24: cost = -7.82813895796069, shots_used = 45762
Step 25: cost = -7.796501914990248, shots_used = 51162
Step 26: cost = -7.871130124788932, shots_used = 56466
Step 27: cost = -7.866190872563943, shots_used = 62010
Step 28: cost = -7.780118268373553, shots_used = 68250
Step 29: cost = -7.843565291223448, shots_used = 74946
Step 30: cost = -7.840084824878835, shots_used = 81762
Step 31: cost = -7.863430860462219, shots_used = 88962
Step 32: cost = -7.863400771365601, shots_used = 96786
Step 33: cost = -7.828392469226825, shots_used = 104730
Step 34: cost = -7.845758777555817, shots_used = 114532
Step 35: cost = -7.862280441095794, shots_used = 122908
Step 36: cost = -7.866212335569502, shots_used = 131836
Step 37: cost = -7.859430128177042, shots_used = 140500
Step 38: cost = -7.856087432905534, shots_used = 150076
Step 39: cost = -7.850323433779115, shots_used = 159676
Step 40: cost = -7.834403598788763, shots_used = 170116
Step 41: cost = -7.849769789802028, shots_used = 181300
Step 42: cost = -7.86693841353118, shots_used = 192700
Step 43: cost = -7.865653895759861, shots_used = 204460
Step 44: cost = -7.853522061269157, shots_used = 217900
Step 45: cost = -7.885272132729725, shots_used = 231748
Step 46: cost = -7.88224395467864, shots_used = 245644
Step 47: cost = -7.884376349618622, shots_used = 259852
Step 48: cost = -7.8808911781003825, shots_used = 275164
Step 49: cost = -7.881035167671664, shots_used = 292444
Step 50: cost = -7.881931152903569, shots_used = 310300
Step 51: cost = -7.873486288144938, shots_used = 329452
Step 52: cost = -7.842973314288795, shots_used = 348532
Step 53: cost = -7.87101794797729, shots_used = 368644
Step 54: cost = -7.880857865087542, shots_used = 388828
Step 55: cost = -7.884163217633474, shots_used = 409132
Step 56: cost = -7.866452206380498, shots_used = 429076
Step 57: cost = -7.876255345278057, shots_used = 451468
Step 58: cost = -7.87369984074766, shots_used = 475348
Step 59: cost = -7.890243502630163, shots_used = 501460
2400
Step 0: cost = -2.03376839972733 shots_used = 2400
Step 1: cost = -3.0397515887713897 shots_used = 4800
Step 2: cost = -3.8459175082365666 shots_used = 7200
Step 3: cost = -4.505506895275778 shots_used = 9600
Step 4: cost = -5.0488106623708084 shots_used = 12000
Step 5: cost = -5.482162129547712 shots_used = 14400
Step 6: cost = -5.83880726147689 shots_used = 16800
Step 7: cost = -6.143933494222608 shots_used = 19200
Step 8: cost = -6.412317130720796 shots_used = 21600
Step 9: cost = -6.6534666682698 shots_used = 24000
Step 10: cost = -6.86746547637287 shots_used = 26400
Step 11: cost = -7.057043661341395 shots_used = 28800
Step 12: cost = -7.219548494479429 shots_used = 31200
Step 13: cost = -7.3445177518694456 shots_used = 33600
Step 14: cost = -7.435753942420535 shots_used = 36000
Step 15: cost = -7.497138548636965 shots_used = 38400
Step 16: cost = -7.529946318655265 shots_used = 40800
Step 17: cost = -7.537070813893377 shots_used = 43200
Step 18: cost = -7.525225697166624 shots_used = 45600
Step 19: cost = -7.5048251159723405 shots_used = 48000
Step 20: cost = -7.481487171246212 shots_used = 50400
Step 21: cost = -7.461106527571478 shots_used = 52800
Step 22: cost = -7.4490325775024075 shots_used = 55200
Step 23: cost = -7.444817343084735 shots_used = 57600
Step 24: cost = -7.4494913586937574 shots_used = 60000
Step 25: cost = -7.462969617594349 shots_used = 62400
Step 26: cost = -7.484518392550573 shots_used = 64800
Step 27: cost = -7.509533957688121 shots_used = 67200
Step 28: cost = -7.535240804873656 shots_used = 69600
Step 29: cost = -7.560642729685874 shots_used = 72000
Step 30: cost = -7.586205677180162 shots_used = 74400
Step 31: cost = -7.61260475402048 shots_used = 76800
Step 32: cost = -7.637117815005769 shots_used = 79200
Step 33: cost = -7.661716123608457 shots_used = 81600
Step 34: cost = -7.6852319189727165 shots_used = 84000
Step 35: cost = -7.708583289744081 shots_used = 86400
Step 36: cost = -7.729551671925802 shots_used = 88800
Step 37: cost = -7.7462558125604595 shots_used = 91200
Step 38: cost = -7.758965992155235 shots_used = 93600
Step 39: cost = -7.764889692835303 shots_used = 96000
Step 40: cost = -7.770298814247658 shots_used = 98400
Step 41: cost = -7.771938304013664 shots_used = 100800
Step 42: cost = -7.771490419427766 shots_used = 103200
Step 43: cost = -7.771665932203987 shots_used = 105600
Step 44: cost = -7.771775966399097 shots_used = 108000
Step 45: cost = -7.772019786144459 shots_used = 110400
Step 46: cost = -7.774409408800273 shots_used = 112800
Step 47: cost = -7.777544198411677 shots_used = 115200
Step 48: cost = -7.78057842461007 shots_used = 117600
Step 49: cost = -7.7865146226898805 shots_used = 120000
Step 50: cost = -7.793839215454196 shots_used = 122400
Step 51: cost = -7.802144039740554 shots_used = 124800
Step 52: cost = -7.809859012081808 shots_used = 127200
Step 53: cost = -7.818330164675909 shots_used = 129600
Step 54: cost = -7.826930993976666 shots_used = 132000
Step 55: cost = -7.834969848723968 shots_used = 134400
Step 56: cost = -7.842454395123664 shots_used = 136800
Step 57: cost = -7.849335152675151 shots_used = 139200
Step 58: cost = -7.853951071633944 shots_used = 141600
Step 59: cost = -7.858296868696565 shots_used = 144000
Step 60: cost = -7.862867672169834 shots_used = 146400
Step 61: cost = -7.865540080202736 shots_used = 148800
Step 62: cost = -7.867577632485199 shots_used = 151200
Step 63: cost = -7.869035010771334 shots_used = 153600
Step 64: cost = -7.870496374034538 shots_used = 156000
Step 65: cost = -7.871678720443278 shots_used = 158400
Step 66: cost = -7.872542373444428 shots_used = 160800
Step 67: cost = -7.873739299675017 shots_used = 163200
Step 68: cost = -7.874314293738313 shots_used = 165600
Step 69: cost = -7.875793149514538 shots_used = 168000
Step 70: cost = -7.877051911492931 shots_used = 170400
Step 71: cost = -7.878207264678217 shots_used = 172800
Step 72: cost = -7.879198045006914 shots_used = 175200
Step 73: cost = -7.880726987471535 shots_used = 177600
Step 74: cost = -7.882055795432435 shots_used = 180000
Step 75: cost = -7.88215282515028 shots_used = 182400
Step 76: cost = -7.881947191378357 shots_used = 184800
Step 77: cost = -7.881566349945106 shots_used = 187200
Step 78: cost = -7.881659168988012 shots_used = 189600
Step 79: cost = -7.881276797156975 shots_used = 192000
Step 80: cost = -7.879976174007023 shots_used = 194400
Step 81: cost = -7.878714918643873 shots_used = 196800
Step 82: cost = -7.877964404670651 shots_used = 199200
Step 83: cost = -7.8771022016203665 shots_used = 201600
Step 84: cost = -7.875562772172711 shots_used = 204000
Step 85: cost = -7.875602350174969 shots_used = 206400
Step 86: cost = -7.877141380119034 shots_used = 208800
Step 87: cost = -7.87925788505365 shots_used = 211200
Step 88: cost = -7.881144761009377 shots_used = 213600
Step 89: cost = -7.882250363744701 shots_used = 216000
Step 90: cost = -7.881748113564451 shots_used = 218400
Step 91: cost = -7.883533319932514 shots_used = 220800
Step 92: cost = -7.884779159318079 shots_used = 223200
Step 93: cost = -7.8868911005436555 shots_used = 225600
Step 94: cost = -7.888524224480213 shots_used = 228000
Step 95: cost = -7.888123287772768 shots_used = 230400
Step 96: cost = -7.8867800801467896 shots_used = 232800
Step 97: cost = -7.8853107450636415 shots_used = 235200
Step 98: cost = -7.883507674089132 shots_used = 237600
Step 99: cost = -7.881351067687096 shots_used = 240000
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step 0: cost = -0.47971271815214855 shots used = 0
Step 1: cost = -1.6879973520840041 shots used = 8000
Step 2: cost = -2.437928256197112 shots used = 16000
Step 3: cost = -2.9300968884147647 shots used = 24000
Step 4: cost = -3.7779069617997116 shots used = 32000
Step 5: cost = -3.8889841568955115 shots used = 40000
Step 6: cost = -4.508059711766957 shots used = 48000
Step 7: cost = -4.71114219758592 shots used = 56000
Step 8: cost = -4.984457128293103 shots used = 64000
Step 9: cost = -5.597084424095087 shots used = 72000
Step 10: cost = -5.456976403687039 shots used = 80000
Step 11: cost = -5.736752824027413 shots used = 88000
Step 12: cost = -6.220317925041974 shots used = 96000
Step 13: cost = -6.45162161927903 shots used = 104000
Step 14: cost = -6.563539211112225 shots used = 112000
Step 15: cost = -6.487339064303318 shots used = 120000
Step 16: cost = -6.69261841162329 shots used = 128000
Step 17: cost = -6.909230576241427 shots used = 136000
Step 18: cost = -7.05156660241221 shots used = 144000
Step 19: cost = -7.163688069859358 shots used = 152000
Step 20: cost = -7.191791478058647 shots used = 160000
Step 21: cost = -7.191694602776715 shots used = 168000
Step 22: cost = -7.430122007574104 shots used = 176000
Step 23: cost = -7.245621601209081 shots used = 184000
Step 24: cost = -7.539044265851978 shots used = 192000
Step 25: cost = -7.532847998808006 shots used = 200000
Step 26: cost = -7.44257222073886 shots used = 208000
Step 27: cost = -7.439951968648378 shots used = 216000
Step 28: cost = -7.734568855081575 shots used = 224000
Step 29: cost = -7.618221322585628 shots used = 232000
Step 30: cost = -7.651544920606065 shots used = 240000
Step 31: cost = -7.5069088885777155 shots used = 248000
Step 32: cost = -7.780301321189146 shots used = 256000
Step 33: cost = -7.4456447455856445 shots used = 264000
Step 34: cost = -7.403560444278863 shots used = 272000
Step 35: cost = -7.666718876831026 shots used = 280000
Step 36: cost = -7.7178910518866415 shots used = 288000
Step 37: cost = -7.375680885292107 shots used = 296000
Step 38: cost = -7.665568049279896 shots used = 304000
Step 39: cost = -7.568101693343673 shots used = 312000
Step 40: cost = -7.524188200359864 shots used = 320000
Step 41: cost = -7.525528734255245 shots used = 328000
Step 42: cost = -7.57734861403185 shots used = 336000
Step 43: cost = -7.76844833198197 shots used = 344000
Step 44: cost = -7.797619087079373 shots used = 352000
Step 45: cost = -7.879148884805528 shots used = 360000
Step 46: cost = -7.744030492750696 shots used = 368000
Step 47: cost = -7.6484739221198765 shots used = 376000
Step 48: cost = -7.679623095926702 shots used = 384000
Step 49: cost = -7.607476988501242 shots used = 392000
Step 50: cost = -7.856041856821188 shots used = 400000
Step 51: cost = -7.644473030321983 shots used = 408000
Step 52: cost = -7.593159311741706 shots used = 416000
Step 53: cost = -7.606939212888227 shots used = 424000
Step 54: cost = -7.621128949485829 shots used = 432000
Step 55: cost = -7.743568287057952 shots used = 440000
Step 56: cost = -7.6325929460598525 shots used = 448000
Step 57: cost = -7.718256562367575 shots used = 456000
Step 58: cost = -7.861601938446393 shots used = 464000
Step 59: cost = -7.666115854972354 shots used = 472000
Step 60: cost = -7.644148944168839 shots used = 480000
Step 61: cost = -7.771569192260795 shots used = 488000
Step 62: cost = -7.776898446282362 shots used = 496000
Step 63: cost = -7.711006891533269 shots used = 504000
Step 64: cost = -7.748650044666392 shots used = 512000
Step 65: cost = -7.690723991927554 shots used = 520000
Step 66: cost = -7.694117031088106 shots used = 528000
Step 67: cost = -7.793250125674997 shots used = 536000
Step 68: cost = -7.926049735334674 shots used = 544000
Step 69: cost = -7.686292326080605 shots used = 552000
Step 70: cost = -7.745774212716911 shots used = 560000
Step 71: cost = -7.625346751584894 shots used = 568000
Step 72: cost = -7.846664469958039 shots used = 576000
Step 73: cost = -7.860275655123486 shots used = 584000
Step 74: cost = -7.593043619614097 shots used = 592000
Step 75: cost = -7.7969799318129045 shots used = 600000
Step 76: cost = -7.837545360539077 shots used = 608000
Step 77: cost = -7.845253964960701 shots used = 616000
Step 78: cost = -7.941652692590529 shots used = 624000
Step 79: cost = -7.967099906804574 shots used = 632000
Step 80: cost = -7.803163356121793 shots used = 640000
Step 81: cost = -7.665600401510319 shots used = 648000
Step 82: cost = -8.09158124610039 shots used = 656000
Step 83: cost = -7.774883584668083 shots used = 664000
Step 84: cost = -7.758175214036924 shots used = 672000
Step 85: cost = -7.9169924228411865 shots used = 680000
Step 86: cost = -7.670199051467696 shots used = 688000
Step 87: cost = -8.085682024006845 shots used = 696000
Step 88: cost = -7.8433919424579095 shots used = 704000
Step 89: cost = -7.755236580472145 shots used = 712000
Step 90: cost = -7.847624689390126 shots used = 720000
Step 91: cost = -8.122239105086607 shots used = 728000
Step 92: cost = -7.922374192271718 shots used = 736000
Step 93: cost = -7.904676929818973 shots used = 744000
Step 94: cost = -7.909417248833883 shots used = 752000
Step 95: cost = -8.06033491620787 shots used = 760000
Step 96: cost = -7.765636196903123 shots used = 768000
Step 97: cost = -7.801666008865329 shots used = 776000
Step 98: cost = -8.066513329432457 shots used = 784000
Step 99: cost = -7.8942080196569675 shots used = 792000
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step 0: cost = -0.38250000000000006 shots used = 0
Step 1: cost = -1.7450000000000006 shots used = 8000
Step 2: cost = -2.54875 shots used = 16000
Step 3: cost = -2.91 shots used = 24000
Step 4: cost = -3.4762500000000003 shots used = 32000
Step 5: cost = -4.08875 shots used = 40000
Step 6: cost = -4.586250000000001 shots used = 48000
Step 7: cost = -4.805 shots used = 56000
Step 8: cost = -4.925 shots used = 64000
Step 9: cost = -5.385000000000001 shots used = 72000
Step 10: cost = -5.4725 shots used = 80000
Step 11: cost = -5.63875 shots used = 88000
Step 12: cost = -5.796250000000001 shots used = 96000
Step 13: cost = -6.308750000000001 shots used = 104000
Step 14: cost = -6.2524999999999995 shots used = 112000
Step 15: cost = -6.706249999999999 shots used = 120000
Step 16: cost = -6.711250000000001 shots used = 128000
Step 17: cost = -6.803749999999999 shots used = 136000
Step 18: cost = -6.94375 shots used = 144000
Step 19: cost = -7.2837499999999995 shots used = 152000
Step 20: cost = -7.4 shots used = 160000
Step 21: cost = -7.38375 shots used = 168000
Step 22: cost = -7.40125 shots used = 176000
Step 23: cost = -7.4775 shots used = 184000
Step 24: cost = -7.58 shots used = 192000
Step 25: cost = -7.623749999999999 shots used = 200000
Step 26: cost = -7.49625 shots used = 208000
Step 27: cost = -7.58375 shots used = 216000
Step 28: cost = -7.6312500000000005 shots used = 224000
Step 29: cost = -7.13375 shots used = 232000
Step 30: cost = -7.47 shots used = 240000
Step 31: cost = -7.6075 shots used = 248000
Step 32: cost = -7.34875 shots used = 256000
Step 33: cost = -7.6525 shots used = 264000
Step 34: cost = -7.572500000000001 shots used = 272000
Step 35: cost = -7.390000000000001 shots used = 280000
Step 36: cost = -7.76375 shots used = 288000
Step 37: cost = -7.49 shots used = 296000
Step 38: cost = -7.61625 shots used = 304000
Step 39: cost = -7.695 shots used = 312000
Step 40: cost = -7.702499999999999 shots used = 320000
Step 41: cost = -7.59625 shots used = 328000
Step 42: cost = -7.733750000000001 shots used = 336000
Step 43: cost = -7.6875 shots used = 344000
Step 44: cost = -7.75875 shots used = 352000
Step 45: cost = -7.796250000000001 shots used = 360000
Step 46: cost = -7.7387500000000005 shots used = 368000
Step 47: cost = -7.92375 shots used = 376000
Step 48: cost = -7.6225 shots used = 384000
Step 49: cost = -7.8425 shots used = 392000
Step 50: cost = -7.74 shots used = 400000
Step 51: cost = -7.661250000000001 shots used = 408000
Step 52: cost = -7.786250000000001 shots used = 416000
Step 53: cost = -7.78875 shots used = 424000
Step 54: cost = -7.62375 shots used = 432000
Step 55: cost = -7.9375 shots used = 440000
Step 56: cost = -7.71625 shots used = 448000
Step 57: cost = -7.72375 shots used = 456000
Step 58: cost = -7.741250000000001 shots used = 464000
Step 59: cost = -7.811249999999999 shots used = 472000
Step 60: cost = -7.89 shots used = 480000
Step 61: cost = -7.74 shots used = 488000
Step 62: cost = -7.751250000000001 shots used = 496000
Step 63: cost = -7.71875 shots used = 504000
Step 64: cost = -7.695 shots used = 512000
Step 65: cost = -7.7325 shots used = 520000
Step 66: cost = -7.819999999999999 shots used = 528000
Step 67: cost = -7.981249999999999 shots used = 536000
Step 68: cost = -7.8 shots used = 544000
Step 69: cost = -7.89 shots used = 552000
Step 70: cost = -7.7125 shots used = 560000
Step 71: cost = -7.993750000000001 shots used = 568000
Step 72: cost = -7.772499999999999 shots used = 576000
Step 73: cost = -8.01125 shots used = 584000
Step 74: cost = -8.116249999999999 shots used = 592000
Step 75: cost = -7.9662500000000005 shots used = 600000
Step 76: cost = -7.7125 shots used = 608000
Step 77: cost = -7.8925 shots used = 616000
Step 78: cost = -7.967499999999999 shots used = 624000
Step 79: cost = -7.91375 shots used = 632000
Step 80: cost = -7.797499999999999 shots used = 640000
Step 81: cost = -7.9975000000000005 shots used = 648000
Step 82: cost = -7.99 shots used = 656000
Step 83: cost = -7.7124999999999995 shots used = 664000
Step 84: cost = -7.76875 shots used = 672000
Step 85: cost = -7.62 shots used = 680000
Step 86: cost = -7.822500000000001 shots used = 688000
Step 87: cost = -7.74625 shots used = 696000
Step 88: cost = -7.9137499999999985 shots used = 704000
Step 89: cost = -7.86125 shots used = 712000
Step 90: cost = -7.975 shots used = 720000
Step 91: cost = -7.89375 shots used = 728000
Step 92: cost = -8.1075 shots used = 736000
Step 93: cost = -7.775 shots used = 744000
Step 94: cost = -7.8999999999999995 shots used = 752000
Step 95: cost = -7.85625 shots used = 760000
Step 96: cost = -7.925000000000001 shots used = 768000
Step 97: cost = -8.0 shots used = 776000
Step 98: cost = -7.825000000000001 shots used = 784000
Step 99: cost = -7.999999999999999 shots used = 792000
Step 0: cost = -5.976611864639144, shots_used = 240
Step 1: cost = -3.9696542358660754, shots_used = 288
Step 2: cost = -4.960189727105252, shots_used = 360
Step 3: cost = -4.580003760087763, shots_used = 456
Step 4: cost = -2.2302167491286937, shots_used = 552
Step 5: cost = -3.639026220963565, shots_used = 696
Step 6: cost = -6.407579837465837, shots_used = 1050
Step 7: cost = -7.436653687431254, shots_used = 1578
Step 8: cost = -7.2596043217789035, shots_used = 2250
Step 9: cost = -7.062132684694291, shots_used = 2970
Step 10: cost = -7.553938182352898, shots_used = 3738
Step 11: cost = -7.530120251217973, shots_used = 4866
Step 12: cost = -7.620064018172074, shots_used = 6474
Step 13: cost = -7.749105026853707, shots_used = 8288
Step 14: cost = -7.758466910010546, shots_used = 10388
Step 15: cost = -7.547668090788592, shots_used = 12404
Step 16: cost = -7.802606000681808, shots_used = 14660
Step 17: cost = -7.8193751054958875, shots_used = 17180
Step 18: cost = -7.813893056373781, shots_used = 19700
Step 19: cost = -7.818976697763794, shots_used = 22796
Step 20: cost = -7.847655565015215, shots_used = 26372
Step 21: cost = -7.854512274045721, shots_used = 30810
Step 22: cost = -7.855665819254091, shots_used = 35538
Step 23: cost = -7.843276666680191, shots_used = 40770
Step 24: cost = -7.828138957960686, shots_used = 45762
Step 25: cost = -7.796501914990251, shots_used = 51162
Step 26: cost = -7.871130124788932, shots_used = 56466
Step 27: cost = -7.8661908725639424, shots_used = 62010
Step 28: cost = -7.780118268373547, shots_used = 68250
Step 29: cost = -7.84356529122345, shots_used = 74946
Step 30: cost = -7.840084824878836, shots_used = 81762
Step 31: cost = -7.8634308604622145, shots_used = 88962
Step 32: cost = -7.863400771365604, shots_used = 96786
Step 33: cost = -7.828392469226824, shots_used = 104730
Step 34: cost = -7.845758777555815, shots_used = 114532
Step 35: cost = -7.862280441095794, shots_used = 122908
Step 36: cost = -7.866212335569504, shots_used = 131836
Step 37: cost = -7.859430128177041, shots_used = 140500
Step 38: cost = -7.856087432905531, shots_used = 150076
Step 39: cost = -7.85032343377911, shots_used = 159676
Step 40: cost = -7.834403598788761, shots_used = 170116
Step 41: cost = -7.849769789802026, shots_used = 181300
Step 42: cost = -7.866938413531174, shots_used = 192700
Step 43: cost = -7.865653895759863, shots_used = 204460
Step 44: cost = -7.853522061269166, shots_used = 217900
Step 45: cost = -7.885272132729721, shots_used = 231748
Step 46: cost = -7.8822439546786445, shots_used = 245644
Step 47: cost = -7.8843763496186225, shots_used = 259852
Step 48: cost = -7.880891178100387, shots_used = 275164
Step 49: cost = -7.881035167671659, shots_used = 292444
Step 50: cost = -7.881931152903572, shots_used = 310300
Step 51: cost = -7.873486288144935, shots_used = 329452
Step 52: cost = -7.8429733142888, shots_used = 348532
Step 53: cost = -7.8710179479772915, shots_used = 368644
Step 54: cost = -7.8808578650875445, shots_used = 388828
Step 55: cost = -7.884163217633472, shots_used = 409132
Step 56: cost = -7.866452206380503, shots_used = 429076
Step 57: cost = -7.876255345278055, shots_used = 451468
Step 58: cost = -7.873699840747662, shots_used = 475348
Step 59: cost = -7.8902435026301605, shots_used = 501460
2400
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step 0: cost = -2.0337683997273297 shots_used = 2400
Step 1: cost = -3.0397515887713924 shots_used = 4800
Step 2: cost = -3.845917508236566 shots_used = 7200
Step 3: cost = -4.505506895275779 shots_used = 9600
Step 4: cost = -5.048810662370808 shots_used = 12000
Step 5: cost = -5.482162129547708 shots_used = 14400
Step 6: cost = -5.838807261476887 shots_used = 16800
Step 7: cost = -6.143933494222609 shots_used = 19200
Step 8: cost = -6.412317130720797 shots_used = 21600
Step 9: cost = -6.653466668269801 shots_used = 24000
Step 10: cost = -6.86746547637287 shots_used = 26400
Step 11: cost = -7.057043661341393 shots_used = 28800
Step 12: cost = -7.219548494479426 shots_used = 31200
Step 13: cost = -7.3445177518694456 shots_used = 33600
Step 14: cost = -7.435753942420526 shots_used = 36000
Step 15: cost = -7.497138548636965 shots_used = 38400
Step 16: cost = -7.5299463186552655 shots_used = 40800
Step 17: cost = -7.537070813893375 shots_used = 43200
Step 18: cost = -7.525225697166626 shots_used = 45600
Step 19: cost = -7.504825115972339 shots_used = 48000
Step 20: cost = -7.481487171246211 shots_used = 50400
Step 21: cost = -7.461106527571477 shots_used = 52800
Step 22: cost = -7.449032577502404 shots_used = 55200
Step 23: cost = -7.444817343084729 shots_used = 57600
Step 24: cost = -7.4494913586937495 shots_used = 60000
Step 25: cost = -7.462969617594352 shots_used = 62400
Step 26: cost = -7.484518392550574 shots_used = 64800
Step 27: cost = -7.509533957688123 shots_used = 67200
Step 28: cost = -7.535240804873657 shots_used = 69600
Step 29: cost = -7.560642729685871 shots_used = 72000
Step 30: cost = -7.586205677180159 shots_used = 74400
Step 31: cost = -7.61260475402048 shots_used = 76800
Step 32: cost = -7.637117815005766 shots_used = 79200
Step 33: cost = -7.661716123608455 shots_used = 81600
Step 34: cost = -7.685231918972718 shots_used = 84000
Step 35: cost = -7.708583289744083 shots_used = 86400
Step 36: cost = -7.729551671925802 shots_used = 88800
Step 37: cost = -7.746255812560461 shots_used = 91200
Step 38: cost = -7.758965992155234 shots_used = 93600
Step 39: cost = -7.7648896928353 shots_used = 96000
Step 40: cost = -7.770298814247661 shots_used = 98400
Step 41: cost = -7.771938304013664 shots_used = 100800
Step 42: cost = -7.771490419427762 shots_used = 103200
Step 43: cost = -7.771665932203989 shots_used = 105600
Step 44: cost = -7.771775966399092 shots_used = 108000
Step 45: cost = -7.772019786144455 shots_used = 110400
Step 46: cost = -7.77440940880027 shots_used = 112800
Step 47: cost = -7.777544198411681 shots_used = 115200
Step 48: cost = -7.780578424610069 shots_used = 117600
Step 49: cost = -7.786514622689886 shots_used = 120000
Step 50: cost = -7.793839215454198 shots_used = 122400
Step 51: cost = -7.802144039740554 shots_used = 124800
Step 52: cost = -7.8098590120818105 shots_used = 127200
Step 53: cost = -7.818330164675915 shots_used = 129600
Step 54: cost = -7.826930993976665 shots_used = 132000
Step 55: cost = -7.834969848723968 shots_used = 134400
Step 56: cost = -7.842454395123669 shots_used = 136800
Step 57: cost = -7.849335152675147 shots_used = 139200
Step 58: cost = -7.853951071633944 shots_used = 141600
Step 59: cost = -7.858296868696568 shots_used = 144000
Step 60: cost = -7.862867672169832 shots_used = 146400
Step 61: cost = -7.86554008020274 shots_used = 148800
Step 62: cost = -7.8675776324852045 shots_used = 151200
Step 63: cost = -7.869035010771336 shots_used = 153600
Step 64: cost = -7.870496374034538 shots_used = 156000
Step 65: cost = -7.871678720443283 shots_used = 158400
Step 66: cost = -7.872542373444427 shots_used = 160800
Step 67: cost = -7.873739299675018 shots_used = 163200
Step 68: cost = -7.874314293738307 shots_used = 165600
Step 69: cost = -7.875793149514543 shots_used = 168000
Step 70: cost = -7.877051911492935 shots_used = 170400
Step 71: cost = -7.878207264678214 shots_used = 172800
Step 72: cost = -7.879198045006914 shots_used = 175200
Step 73: cost = -7.880726987471537 shots_used = 177600
Step 74: cost = -7.88205579543243 shots_used = 180000
Step 75: cost = -7.88215282515028 shots_used = 182400
Step 76: cost = -7.881947191378358 shots_used = 184800
Step 77: cost = -7.881566349945112 shots_used = 187200
Step 78: cost = -7.881659168988009 shots_used = 189600
Step 79: cost = -7.881276797156975 shots_used = 192000
Step 80: cost = -7.879976174007026 shots_used = 194400
Step 81: cost = -7.878714918643873 shots_used = 196800
Step 82: cost = -7.877964404670646 shots_used = 199200
Step 83: cost = -7.877102201620369 shots_used = 201600
Step 84: cost = -7.875562772172705 shots_used = 204000
Step 85: cost = -7.87560235017497 shots_used = 206400
Step 86: cost = -7.877141380119032 shots_used = 208800
Step 87: cost = -7.87925788505365 shots_used = 211200
Step 88: cost = -7.881144761009377 shots_used = 213600
Step 89: cost = -7.882250363744703 shots_used = 216000
Step 90: cost = -7.8817481135644485 shots_used = 218400
Step 91: cost = -7.883533319932512 shots_used = 220800
Step 92: cost = -7.884779159318077 shots_used = 223200
Step 93: cost = -7.886891100543656 shots_used = 225600
Step 94: cost = -7.88852422448021 shots_used = 228000
Step 95: cost = -7.888123287772764 shots_used = 230400
Step 96: cost = -7.8867800801467896 shots_used = 232800
 </code>
 </pre>
 </details>

---

## 3. tutorial_gaussian_transformation.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_gaussian_transformation.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost after step     1: 0.999118
Cost after step     2: 0.998273
Cost after step     3: 0.996618
Cost after step     4: 0.993382
Cost after step     5: 0.987074
Cost after step     6: 0.974837
Cost after step     7: 0.951332
Cost after step     8: 0.907043
Cost after step     9: 0.826649
Cost after step    10: 0.690812
Cost after step    11: 0.490303
Cost after step    12: 0.258845
Cost after step    13: 0.083224
Cost after step    14: 0.013179
Cost after step    15: 0.001001
Cost after step    16: 0.000049
Cost after step    17: 0.000002
Cost after step    18: 0.000000
Cost after step    19: 0.000000
Cost after step    20: 0.000000
Optimized mag_alpha:0.999994
Optimized phase_alpha:0.020000
Optimized phi:0.005000
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_gaussian_transformation.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
  UserWarning,
Cost after step     1: 0.999118
Cost after step     2: 0.998273
Cost after step     3: 0.996618
Cost after step     4: 0.993382
Cost after step     5: 0.987074
Cost after step     6: 0.974837
Cost after step     7: 0.951332
Cost after step     8: 0.907043
Cost after step     9: 0.826649
Cost after step    10: 0.690812
Cost after step    11: 0.490303
Cost after step    12: 0.258845
Cost after step    13: 0.083224
Cost after step    14: 0.013179
Cost after step    15: 0.001001
Cost after step    16: 0.000049
Cost after step    17: 0.000002
Cost after step    18: 0.000000
Cost after step    19: 0.000000
Cost after step    20: 0.000000
Optimized mag_alpha:0.999994
 </code>
 </pre>
 </details>

---

## 4. tutorial_rotoselect.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rotoselect.html):

```
Optimal generators are: ['Y', 'X']
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_rotoselect.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
```

---

## 5. tutorial_gbs.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_gbs.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_gbs.py:165: UserWarning: 'Interferometer' is deprecated and will be renamed 'InterferometerUnitary'
(10, 10, 10, 10)
|0000>: 0.17637844761413501
|1100>: 0.034732936494202823
|0101>: 0.011870900427255577
|1111>: 0.005957399165336117
|2000>: 0.02957384308320544
[[ 0.19343159-0.54582922j  0.43418269-0.09169615j]
 [ 0.43418269-0.09169615j -0.27554025-0.46222197j]]
0.1763784476141347
0.17637844761413501
0.03473293649420271
0.034732936494202823
0.011870900427255558
0.011870900427255577
0.005957399165336081
0.005957399165336117
0.02957384308320539
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_gbs.html):

```
(10, 10, 10, 10)
|0000>: 0.17637844761413501
|1100>: 0.034732936494202844
|0101>: 0.011870900427255577
|1111>: 0.005957399165336121
|2000>: 0.029573843083205452
[[ 0.19343159-0.54582922j  0.43418269-0.09169615j]
 [ 0.43418269-0.09169615j -0.27554025-0.46222197j]]
0.1763784476141347
0.17637844761413501
0.03473293649420271
0.034732936494202844
0.011870900427255547
0.011870900427255577
0.005957399165336084
0.005957399165336121
0.02957384308320539
0.029573843083205452
```

---

## 6. tutorial_noisy_circuits.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_noisy_circuits.html):

```
Step: 5    Cost: 0.07733960999999957
Step: 10    Cost: 0.0773396099969988
Step: 15    Cost: 0.07733959171203489
Step: 20    Cost: 0.07722827121891838
Step: 25    Cost: 0.0017923029380396919
Step: 30    Cost: 3.0199179590479204e-07
Step: 34    Cost: 5.228404765345524e-10
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_noisy_circuits.html):

```
Step: 5    Cost: 0.07733960999999988
Step: 10    Cost: 0.07733960999863909
Step: 15    Cost: 0.07733960170319246
Step: 20    Cost: 0.07728907281668594
Step: 25    Cost: 0.006192562764640602
Step: 30    Cost: 6.427645677603198e-07
Step: 34    Cost: 1.1072988376257744e-09
```

---

## 7. tutorial_jax_transformations.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
Result: DeviceArray(0.99244501, dtype=float64)
No jit time: 0.0118 seconds
First run time: 0.0550 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
Result: DeviceArray(0.99244503, dtype=float64)
No jit time: 0.0070 seconds
First run time: 0.0522 seconds
```

---

## 8. tutorial_qgrnn.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost at Step 0: -0.9803638573791904
Weights at Step 0: [-0.22604317  0.4388776   0.85859736  0.69736712  0.09417674 -0.02437703]
Bias at Step 0: [-0.23885902 -0.21393414  0.12811164  0.45038514]
---------------------------------------------
Cost at Step 5: -0.9806500098112143
Weights at Step 5: [-1.29827824  1.52426565  1.81163837  1.86438612  1.04288314 -0.98516982]
Bias at Step 5: [-1.27134815 -1.36193505  1.31543373  1.32653424]
---------------------------------------------
Cost at Step 10: -0.9648857984236838
Weights at Step 10: [-1.41068173  1.67469055  1.64410873  2.23403518  0.87027277 -0.84569027]
Bias at Step 10: [-1.28108438 -1.67672193  1.74541519  0.99186816]
---------------------------------------------
Cost at Step 15: -0.9909075076678548
Weights at Step 15: [-0.99423966  1.31509032  0.97714182  2.16814495  0.20032301 -0.2265963 ]
Bias at Step 15: [-0.74022191 -1.53032257  1.76965387  0.16183654]
---------------------------------------------
Cost at Step 20: -0.9966008204823482
Weights at Step 20: [-0.46419217  0.84550568  0.43286203  1.9380202  -0.348272    0.25976054]
Bias at Step 20: [-0.12768221 -1.22298499  1.62611891 -0.4553839 ]
---------------------------------------------
Cost at Step 25: -0.9926133497439015
Weights at Step 25: [-0.09015325  0.52906304  0.38274877  1.70919239 -0.41967469  0.26112821]
Bias at Step 25: [ 0.2326128  -0.93936924  1.45717265 -0.4540624 ]
---------------------------------------------
Cost at Step 30: -0.9984946331046769
Weights at Step 30: [ 0.0123472   0.47133081  0.7428572   1.56943045 -0.11998299 -0.09999201]
Bias at Step 30: [ 0.22093363 -0.77889245  1.33551884 -0.01237726]
---------------------------------------------
Cost at Step 35: -0.9987664875905384
Weights at Step 35: [-0.04974879  0.55838325  1.09528857  1.50649468  0.13247182 -0.39189813]
Bias at Step 35: [ 0.01173716 -0.72349491  1.25413089  0.33198678]
---------------------------------------------
Cost at Step 40: -0.9976572253882188
Weights at Step 40: [-0.14927     0.66157373  1.19794921  1.48297288  0.10734093 -0.3831593 ]
Bias at Step 40: [-0.21739349 -0.72790187  1.18290627  0.29425225]
---------------------------------------------
Cost at Step 45: -0.9996712273475132
Weights at Step 45: [-0.20674402  0.70360037  1.0807707   1.48455205 -0.12668579 -0.15106377]
Bias at Step 45: [-0.35389918 -0.76974032  1.11755141 -0.02878183]
---------------------------------------------
Cost at Step 50: -0.9995584075846883
Weights at Step 50: [-0.21856857  0.69313289  0.9951204   1.51327524 -0.27695939 -0.00304774]
Bias at Step 50: [-0.39501335 -0.83717604  1.08135172 -0.25421503]
---------------------------------------------
Cost at Step 55: -0.9993417338158985
Weights at Step 55: [-0.19958537  0.65456745  1.0540154   1.56588539 -0.22806521 -0.06304728]
Bias at Step 55: [-0.37502663 -0.92029696  1.08093403 -0.2300362 ]
---------------------------------------------
Cost at Step 60: -0.9997702657469879
Weights at Step 60: [-0.15016116  0.59649054  1.17917889  1.61891613 -0.0895     -0.22355593]
Bias at Step 60: [-0.3186739  -0.99107611  1.09766839 -0.08892556]
---------------------------------------------
Cost at Step 65: -0.9996436456556429
Weights at Step 65: [-0.08220889  0.53561352  1.22392446  1.64470734 -0.04968037 -0.29147776]
Bias at Step 65: [-0.26354431 -1.02050635  1.10560821 -0.06357961]
---------------------------------------------
Cost at Step 70: -0.9997829725978746
Weights at Step 70: [-0.02812171  0.50207641  1.18712749  1.64861258 -0.12278163 -0.24907531]
Bias at Step 70: [-0.25578322 -1.01899708  1.10378897 -0.1755329 ]
---------------------------------------------
Cost at Step 75: -0.9998161136786834
Weights at Step 75: [-0.01479171  0.51567978  1.17552912  1.65394365 -0.18664303 -0.21758165]
Bias at Step 75: [-0.32014432 -1.01791815  1.10519351 -0.27848449]
---------------------------------------------
Cost at Step 80: -0.9998843655288555
Weights at Step 80: [-0.01884375  0.54281647  1.22344387  1.66582127 -0.17956552 -0.25497115]
Bias at Step 80: [-0.40362773 -1.02780124  1.10836139 -0.29496652]
---------------------------------------------
Cost at Step 85: -0.999909835033078
Weights at Step 85: [-0.00379234  0.54313665  1.2733362   1.67338414 -0.15161841 -0.30903377]
Bias at Step 85: [-0.45061752 -1.03818189  1.10130953 -0.28564262]
---------------------------------------------
Cost at Step 90: -0.9999044415579313
Weights at Step 90: [ 0.04088165  0.50692558  1.28240681  1.67288196 -0.15109114 -0.33205828]
Bias at Step 90: [-0.4519275  -1.04504232  1.07941985 -0.3132239 ]
---------------------------------------------
Cost at Step 95: -0.9999061222838249
Weights at Step 95: [ 0.08554249  0.46656083  1.26840277  1.6755038  -0.16748469 -0.33495651]
Bias at Step 95: [-0.45081552 -1.0592657   1.05546404 -0.36819017]
---------------------------------------------
Cost at Step 100: -0.9999067038489897
Weights at Step 100: [ 0.10504756  0.45151576  1.27495676  1.68957857 -0.16364088 -0.35685113]
Bias at Step 100: [-0.48320092 -1.0860277   1.04231719 -0.40281373]
---------------------------------------------
Cost at Step 105: -0.9999258608983025
Weights at Step 105: [ 0.11286395  0.45096054  1.29489123  1.70583499 -0.14822871 -0.39062082]
Bias at Step 105: [-0.53290732 -1.11214824  1.03558616 -0.42231884]
---------------------------------------------
Cost at Step 110: -0.9999159642471993
Weights at Step 110: [ 0.13106534  0.44316269  1.30188589  1.7135758  -0.14449277 -0.41395624]
Bias at Step 110: [-0.57365278 -1.12589957  1.02588522 -0.45380079]
---------------------------------------------
Cost at Step 115: -0.9999295051359973
Weights at Step 115: [ 0.16456919  0.42229382  1.29615959  1.71331202 -0.14936482 -0.43044986]
Bias at Step 115: [-0.60049609 -1.12909729  1.01159669 -0.4958413 ]
---------------------------------------------
Cost at Step 120: -0.9999329105144759
Weights at Step 120: [ 0.19394551  0.40471337  1.2991934   1.71575281 -0.1428606  -0.45732438]
Bias at Step 120: [-0.63053342 -1.13507053  1.00043562 -0.52433156]
---------------------------------------------
Cost at Step 125: -0.999967215118185
Weights at Step 125: [ 0.21407342  0.39421863  1.30670262  1.72320127 -0.13102448 -0.48725764]
Bias at Step 125: [-0.6687698  -1.14749235  0.99304941 -0.54780452]
---------------------------------------------
Cost at Step 130: -0.9999498639111292
Weights at Step 130: [ 0.23251895  0.38323767  1.30565485  1.73043603 -0.12634796 -0.50754102]
Bias at Step 130: [-0.70519062 -1.16065043  0.9853521  -0.57984279]
---------------------------------------------
Cost at Step 135: -0.9999460668283626
Weights at Step 135: [ 0.25315     0.36897848  1.30157699  1.73591416 -0.12368359 -0.52479079]
Bias at Step 135: [-0.73738032 -1.17210895  0.97628525 -0.6137189 ]
---------------------------------------------
Cost at Step 140: -0.9999760263168518
Weights at Step 140: [ 0.27319219  0.35494171  1.30465117  1.74054957 -0.11422205 -0.5482196 ]
Bias at Step 140: [-0.76767167 -1.18178289  0.96794962 -0.63664615]
---------------------------------------------
Cost at Step 145: -0.9999620825798219
Weights at Step 145: [ 0.29291312  0.34115888  1.30743903  1.7441703  -0.10602196 -0.56989588]
Bias at Step 145: [-0.79755643 -1.18957764  0.96017139 -0.65982932]
---------------------------------------------
Cost at Step 150: -0.999952816271399
Weights at Step 150: [ 0.31193841  0.32791071  1.30767856  1.747724   -0.10220887 -0.58675715]
Bias at Step 150: [-0.8277568  -1.19660646  0.9533145  -0.68795875]
---------------------------------------------
Cost at Step 155: -0.9999676206975213
Weights at Step 155: [ 0.32929841  0.31576924  1.3101255   1.75246696 -0.0974324  -0.60396184]
Bias at Step 155: [-0.85776514 -1.2043529   0.94822294 -0.71402521]
---------------------------------------------
Cost at Step 160: -0.9999688897065233
Weights at Step 160: [ 0.34617048  0.30341272  1.31549083  1.75789129 -0.09127326 -0.62192276]
Bias at Step 160: [-0.88499311 -1.21222407  0.94414483 -0.73671672]
---------------------------------------------
Cost at Step 165: -0.9999678828349428
Weights at Step 165: [ 0.36424544  0.28904681  1.3183802   1.76299152 -0.08653562 -0.6379972 ]
Bias at Step 165: [-0.91076279 -1.21990516  0.93990101 -0.76174131]
---------------------------------------------
Cost at Step 170: -0.999971851850557
Weights at Step 170: [ 0.38058818  0.27530965  1.31811484  1.76729325 -0.08126922 -0.65326086]
Bias at Step 170: [-0.93762726 -1.22741419  0.93605274 -0.78587399]
---------------------------------------------
Cost at Step 175: -0.999969992884057
Weights at Step 175: [ 0.39531011  0.26293084  1.32005541  1.7712584  -0.0738556  -0.67032688]
Bias at Step 175: [-0.96623738 -1.23426398  0.9328937  -0.80654326]
---------------------------------------------
Cost at Step 180: -0.9999810640341683
Weights at Step 180: [ 0.41111346  0.24939364  1.32150139  1.7744836  -0.06804758 -0.68598269]
Bias at Step 180: [-0.99446201 -1.23978732  0.92974762 -0.82904396]
---------------------------------------------
Cost at Step 185: -0.9999732059499763
Weights at Step 185: [ 0.42663809  0.23569253  1.323561    1.77787964 -0.06331664 -0.7001218 ]
Bias at Step 185: [-1.02027229 -1.24495257  0.9273182  -0.8511393 ]
---------------------------------------------
Cost at Step 190: -0.9999904686011623
Weights at Step 190: [ 0.4407972   0.22274166  1.32741124  1.78217538 -0.0581927  -0.71407865]
Bias at Step 190: [-1.04467037 -1.2507425   0.92600868 -0.87121712]
---------------------------------------------
Cost at Step 195: -0.9999828831078132
Weights at Step 195: [ 0.45318723  0.21066667  1.32928285  1.78607162 -0.05386645 -0.72584502]
Bias at Step 195: [-1.06684646 -1.25630331  0.92524928 -0.88990376]
---------------------------------------------
Cost at Step 200: -0.9999814691774115
Weights at Step 200: [ 0.46483484  0.19879592  1.33066437  1.78942676 -0.04940039 -0.73713324]
Bias at Step 200: [-1.08841247 -1.26129482  0.92474857 -0.90725602]
---------------------------------------------
Cost at Step 205: -0.9999831444629905
Weights at Step 205: [ 0.47584426  0.18719266  1.33154192  1.79202379 -0.04462708 -0.74822003]
Bias at Step 205: [-1.10977785 -1.265427    0.92453063 -0.92310712]
---------------------------------------------
Cost at Step 210: -0.9999790517608405
Weights at Step 210: [ 0.48584547  0.17597834  1.33028455  1.79378517 -0.03978138 -0.75852048]
Bias at Step 210: [-1.13101187 -1.26898531  0.92482346 -0.93780681]
---------------------------------------------
Cost at Step 215: -0.9999847111950139
Weights at Step 215: [ 0.4946749   0.16562713  1.32915967  1.79528287 -0.03462316 -0.76851533]
Bias at Step 215: [-1.15132516 -1.27221485  0.92584491 -0.9503349 ]
---------------------------------------------
Cost at Step 220: -0.9999855424488905
Weights at Step 220: [ 0.50303178  0.15574802  1.32886423  1.79677025 -0.03058318 -0.77711746]
Bias at Step 220: [-1.16955409 -1.27502675  0.92729975 -0.96212831]
---------------------------------------------
Cost at Step 225: -0.9999863772257214
Weights at Step 225: [ 0.51097377  0.14603986  1.32919454  1.79836519 -0.02733105 -0.78463519]
Bias at Step 225: [-1.18628365 -1.27783948  0.92913454 -0.97340009]
---------------------------------------------
Cost at Step 230: -0.9999898484097558
Weights at Step 230: [ 0.51836647  0.13626055  1.32967368  1.80000495 -0.02394918 -0.79187192]
Bias at Step 230: [-1.20276401 -1.28103879  0.93145415 -0.98384786]
---------------------------------------------
Cost at Step 235: -0.9999890657624452
Weights at Step 235: [ 0.52564729  0.12600301  1.32958604  1.80138672 -0.02082854 -0.79856878]
Bias at Step 235: [-1.2195148  -1.28431275  0.93422156 -0.9943289 ]
---------------------------------------------
Cost at Step 240: -0.999988297622605
Weights at Step 240: [ 0.53229253  0.11610906  1.32908336  1.80229993 -0.01749836 -0.80503539]
Bias at Step 240: [-1.23599684 -1.28731013  0.93747248 -1.00346756]
---------------------------------------------
Cost at Step 245: -0.9999882440932367
Weights at Step 245: [ 0.53847247  0.10723924  1.33070351  1.80329887 -0.01467023 -0.81108244]
Bias at Step 245: [-1.25069646 -1.28978222  0.94081211 -1.01123413]
---------------------------------------------
Cost at Step 250: -0.9999868214455344
Weights at Step 250: [ 0.54460039  0.0980518   1.33056073  1.80410968 -0.01354454 -0.81496581]
Bias at Step 250: [-1.26394352 -1.29235223  0.94434983 -1.02022419]
---------------------------------------------
Cost at Step 255: -0.9999884214814982
Weights at Step 255: [ 0.54943379  0.08996909  1.33186446  1.8051759  -0.01056943 -0.82027142]
Bias at Step 255: [-1.27708148 -1.29545081  0.94840617 -1.02592635]
---------------------------------------------
Cost at Step 260: -0.9999905893984564
Weights at Step 260: [ 0.55392271  0.08211047  1.33002746  1.8051647  -0.00912692 -0.82330494]
Bias at Step 260: [-1.28877696 -1.29778044  0.9520992  -1.03219995]
---------------------------------------------
Cost at Step 265: -0.9999894660037792
Weights at Step 265: [ 0.55772452  0.07551996  1.33104998  1.80525639 -0.00672651 -0.82745976]
Bias at Step 265: [-1.29986939 -1.29978116  0.95585094 -1.03589225]
---------------------------------------------
Cost at Step 270: -0.9999873354125405
Weights at Step 270: [ 0.56170933  0.06903136  1.33175345  1.80537894 -0.00624539 -0.82966868]
Bias at Step 270: [-1.30907139 -1.30160912  0.9593855  -1.04092885]
---------------------------------------------
Cost at Step 275: -0.9999910126657188
Weights at Step 275: [ 0.56476346  0.06288961  1.33121415  1.80555827 -0.00483336 -0.83207027]
Bias at Step 275: [-1.3181107  -1.30426179  0.96340533 -1.04444787]
---------------------------------------------
Cost at Step 280: -0.9999911807478907
Weights at Step 280: [ 0.56790545  0.05710628  1.33293783  1.80572737 -0.00337721 -0.83486294]
Bias at Step 280: [-1.32665878 -1.30655911  0.96725682 -1.04712595]
---------------------------------------------
Cost at Step 285: -0.9999870533227306
Weights at Step 285: [ 0.57142315  0.05115447  1.33329561  1.80547158 -0.00412561 -0.83538511]
Bias at Step 285: [-1.33391608 -1.30836527  0.97068764 -1.05192006]
---------------------------------------------
Cost at Step 290: -0.9999918533581701
Weights at Step 290: [ 0.57314839  0.04641148  1.33277082  1.80515273 -0.00185471 -0.83794182]
Bias at Step 290: [-1.3420092  -1.3110848   0.97494496 -1.05274072]
---------------------------------------------
Cost at Step 295: -0.9999907581161834
Weights at Step 295: [ 5.75810409e-01  4.17373794e-02  1.33385607e+00  1.80469466e+00
 -1.37560830e-03 -8.39439431e-01]
Bias at Step 295: [-1.34861394 -1.31269908  0.97854738 -1.05489467]
---------------------------------------------
Target parameters     Learned parameters
Weights
-----------------------------------------
0.56                |  0.5782895244479022
1.24                |  1.3350283296762837
1.67                |  1.8044804399858474
-0.79               | -0.8395497395039527
Bias
-----------------------------------------
-1.44               | -1.3529586931946354
-1.43               | -1.3138918025602149
1.18                |  0.9811173767093428
-0.93               | -1.0579331212003378
Non-Existing Edge Parameters: [0.03795849267891401, -0.0022991817976618463]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Cost at Step 0: -0.9803638573791903
Weights at Step 0: [-0.22604317  0.4388776   0.85859736  0.69736712  0.09417674 -0.02437703]
Bias at Step 0: [-0.23885902 -0.21393414  0.12811164  0.45038514]
---------------------------------------------
Cost at Step 5: -0.9806500098112143
Weights at Step 5: [-1.29827824  1.52426565  1.81163837  1.86438612  1.04288314 -0.98516982]
Bias at Step 5: [-1.27134815 -1.36193505  1.31543373  1.32653424]
---------------------------------------------
Cost at Step 10: -0.9648857984236836
Weights at Step 10: [-1.41068173  1.67469055  1.64410873  2.23403518  0.87027277 -0.84569027]
Bias at Step 10: [-1.28108438 -1.67672193  1.74541519  0.99186816]
---------------------------------------------
Cost at Step 15: -0.9909075076678547
Weights at Step 15: [-0.99423966  1.31509032  0.97714182  2.16814495  0.20032301 -0.2265963 ]
Bias at Step 15: [-0.74022191 -1.53032257  1.76965387  0.16183654]
---------------------------------------------
Cost at Step 20: -0.9966008204823483
Weights at Step 20: [-0.46419217  0.84550568  0.43286203  1.9380202  -0.348272    0.25976054]
Bias at Step 20: [-0.12768221 -1.22298499  1.62611891 -0.4553839 ]
---------------------------------------------
Cost at Step 25: -0.9926133497439016
Weights at Step 25: [-0.09015325  0.52906304  0.38274877  1.70919239 -0.41967469  0.26112821]
Bias at Step 25: [ 0.2326128  -0.93936924  1.45717265 -0.4540624 ]
---------------------------------------------
Cost at Step 30: -0.9984946331046771
Weights at Step 30: [ 0.0123472   0.47133081  0.7428572   1.56943045 -0.11998299 -0.09999201]
Bias at Step 30: [ 0.22093363 -0.77889245  1.33551884 -0.01237726]
---------------------------------------------
Cost at Step 35: -0.9987664875905384
Weights at Step 35: [-0.04974879  0.55838325  1.09528857  1.50649468  0.13247182 -0.39189813]
Bias at Step 35: [ 0.01173716 -0.72349491  1.25413089  0.33198678]
---------------------------------------------
Cost at Step 40: -0.9976572253882187
Weights at Step 40: [-0.14927     0.66157373  1.19794921  1.48297288  0.10734093 -0.3831593 ]
Bias at Step 40: [-0.21739349 -0.72790187  1.18290627  0.29425225]
---------------------------------------------
Cost at Step 45: -0.9996712273475133
Weights at Step 45: [-0.20674402  0.70360037  1.0807707   1.48455205 -0.12668579 -0.15106377]
Bias at Step 45: [-0.35389918 -0.76974032  1.11755141 -0.02878183]
---------------------------------------------
Cost at Step 50: -0.9995584075846882
Weights at Step 50: [-0.21856857  0.69313289  0.9951204   1.51327524 -0.27695939 -0.00304774]
Bias at Step 50: [-0.39501335 -0.83717604  1.08135172 -0.25421503]
---------------------------------------------
Cost at Step 55: -0.9993417338158986
Weights at Step 55: [-0.19958537  0.65456745  1.0540154   1.56588539 -0.22806521 -0.06304728]
Bias at Step 55: [-0.37502663 -0.92029696  1.08093403 -0.2300362 ]
---------------------------------------------
Cost at Step 60: -0.9997702657469877
Weights at Step 60: [-0.15016116  0.59649054  1.17917889  1.61891613 -0.0895     -0.22355593]
Bias at Step 60: [-0.3186739  -0.99107611  1.09766839 -0.08892556]
---------------------------------------------
Cost at Step 65: -0.9996436456556425
Weights at Step 65: [-0.08220889  0.53561352  1.22392446  1.64470734 -0.04968037 -0.29147776]
Bias at Step 65: [-0.26354431 -1.02050635  1.10560821 -0.06357961]
---------------------------------------------
Cost at Step 70: -0.9997829725978745
Weights at Step 70: [-0.02812171  0.50207641  1.18712749  1.64861258 -0.12278163 -0.24907531]
Bias at Step 70: [-0.25578322 -1.01899708  1.10378897 -0.1755329 ]
---------------------------------------------
Cost at Step 75: -0.9998161136786834
Weights at Step 75: [-0.01479171  0.51567978  1.17552912  1.65394365 -0.18664303 -0.21758165]
Bias at Step 75: [-0.32014432 -1.01791815  1.10519351 -0.27848449]
---------------------------------------------
Cost at Step 80: -0.9998843655288554
Weights at Step 80: [-0.01884375  0.54281647  1.22344387  1.66582127 -0.17956552 -0.25497115]
Bias at Step 80: [-0.40362773 -1.02780124  1.10836139 -0.29496652]
---------------------------------------------
Cost at Step 85: -0.9999098350330781
Weights at Step 85: [-0.00379234  0.54313665  1.2733362   1.67338414 -0.15161841 -0.30903377]
Bias at Step 85: [-0.45061752 -1.03818189  1.10130953 -0.28564262]
---------------------------------------------
Cost at Step 90: -0.9999044415579312
Weights at Step 90: [ 0.04088165  0.50692558  1.28240681  1.67288196 -0.15109114 -0.33205828]
Bias at Step 90: [-0.4519275  -1.04504232  1.07941985 -0.3132239 ]
---------------------------------------------
Cost at Step 95: -0.999906122283825
Weights at Step 95: [ 0.08554249  0.46656083  1.26840277  1.6755038  -0.16748469 -0.33495651]
Bias at Step 95: [-0.45081552 -1.0592657   1.05546404 -0.36819017]
---------------------------------------------
Cost at Step 100: -0.9999067038489896
Weights at Step 100: [ 0.10504756  0.45151576  1.27495676  1.68957857 -0.16364088 -0.35685113]
Bias at Step 100: [-0.48320092 -1.0860277   1.04231719 -0.40281373]
---------------------------------------------
Cost at Step 105: -0.9999258608983023
Weights at Step 105: [ 0.11286395  0.45096054  1.29489123  1.70583499 -0.14822871 -0.39062082]
Bias at Step 105: [-0.53290732 -1.11214824  1.03558616 -0.42231884]
---------------------------------------------
Cost at Step 110: -0.9999159642471993
Weights at Step 110: [ 0.13106534  0.44316269  1.30188589  1.7135758  -0.14449277 -0.41395624]
Bias at Step 110: [-0.57365278 -1.12589957  1.02588522 -0.45380079]
---------------------------------------------
Cost at Step 115: -0.9999295051359973
Weights at Step 115: [ 0.16456919  0.42229382  1.29615959  1.71331202 -0.14936482 -0.43044986]
Bias at Step 115: [-0.60049609 -1.12909729  1.01159669 -0.4958413 ]
---------------------------------------------
Cost at Step 120: -0.9999329105144759
Weights at Step 120: [ 0.19394551  0.40471337  1.2991934   1.71575281 -0.1428606  -0.45732438]
Bias at Step 120: [-0.63053342 -1.13507053  1.00043562 -0.52433156]
---------------------------------------------
Cost at Step 125: -0.9999672151181849
Weights at Step 125: [ 0.21407342  0.39421863  1.30670262  1.72320127 -0.13102448 -0.48725764]
Bias at Step 125: [-0.6687698  -1.14749235  0.99304941 -0.54780452]
---------------------------------------------
Cost at Step 130: -0.9999498639111295
Weights at Step 130: [ 0.23251895  0.38323767  1.30565485  1.73043603 -0.12634796 -0.50754102]
Bias at Step 130: [-0.70519062 -1.16065043  0.9853521  -0.57984279]
---------------------------------------------
Cost at Step 135: -0.9999460668283625
Weights at Step 135: [ 0.25315     0.36897848  1.30157699  1.73591416 -0.12368359 -0.52479079]
Bias at Step 135: [-0.73738032 -1.17210895  0.97628525 -0.6137189 ]
---------------------------------------------
Cost at Step 140: -0.9999760263168521
Weights at Step 140: [ 0.27319219  0.35494171  1.30465117  1.74054957 -0.11422205 -0.5482196 ]
Bias at Step 140: [-0.76767167 -1.18178289  0.96794962 -0.63664615]
---------------------------------------------
Cost at Step 145: -0.9999620825798218
Weights at Step 145: [ 0.29291312  0.34115888  1.30743903  1.7441703  -0.10602196 -0.56989588]
Bias at Step 145: [-0.79755643 -1.18957764  0.96017139 -0.65982932]
---------------------------------------------
Cost at Step 150: -0.9999528162713988
Weights at Step 150: [ 0.31193841  0.32791071  1.30767856  1.747724   -0.10220887 -0.58675715]
Bias at Step 150: [-0.8277568  -1.19660646  0.9533145  -0.68795875]
---------------------------------------------
Cost at Step 155: -0.9999676206975212
Weights at Step 155: [ 0.32929841  0.31576924  1.3101255   1.75246696 -0.0974324  -0.60396184]
Bias at Step 155: [-0.85776514 -1.2043529   0.94822294 -0.71402521]
---------------------------------------------
Cost at Step 160: -0.9999688897065233
Weights at Step 160: [ 0.34617048  0.30341272  1.31549083  1.75789129 -0.09127326 -0.62192276]
Bias at Step 160: [-0.88499311 -1.21222407  0.94414483 -0.73671672]
---------------------------------------------
Cost at Step 165: -0.9999678828349426
Weights at Step 165: [ 0.36424544  0.28904681  1.3183802   1.76299152 -0.08653562 -0.6379972 ]
Bias at Step 165: [-0.91076279 -1.21990516  0.93990101 -0.76174131]
---------------------------------------------
Cost at Step 170: -0.9999718518505567
Weights at Step 170: [ 0.38058818  0.27530965  1.31811484  1.76729325 -0.08126922 -0.65326086]
Bias at Step 170: [-0.93762726 -1.22741419  0.93605274 -0.78587399]
---------------------------------------------
Cost at Step 175: -0.9999699928840567
Weights at Step 175: [ 0.39531011  0.26293084  1.32005541  1.7712584  -0.0738556  -0.67032688]
Bias at Step 175: [-0.96623738 -1.23426398  0.9328937  -0.80654326]
---------------------------------------------
Cost at Step 180: -0.9999810640341683
Weights at Step 180: [ 0.41111346  0.24939364  1.32150139  1.7744836  -0.06804758 -0.68598269]
Bias at Step 180: [-0.99446201 -1.23978732  0.92974762 -0.82904396]
---------------------------------------------
Cost at Step 185: -0.9999732059499763
Weights at Step 185: [ 0.42663809  0.23569253  1.323561    1.77787964 -0.06331664 -0.7001218 ]
Bias at Step 185: [-1.02027229 -1.24495257  0.9273182  -0.8511393 ]
---------------------------------------------
Cost at Step 190: -0.9999904686011625
Weights at Step 190: [ 0.4407972   0.22274166  1.32741124  1.78217538 -0.0581927  -0.71407865]
Bias at Step 190: [-1.04467037 -1.2507425   0.92600868 -0.87121712]
---------------------------------------------
Cost at Step 195: -0.999982883107813
Weights at Step 195: [ 0.45318723  0.21066667  1.32928285  1.78607162 -0.05386645 -0.72584502]
Bias at Step 195: [-1.06684646 -1.25630331  0.92524928 -0.88990376]
---------------------------------------------
Cost at Step 200: -0.9999814691774114
Weights at Step 200: [ 0.46483484  0.19879592  1.33066437  1.78942676 -0.04940039 -0.73713324]
Bias at Step 200: [-1.08841247 -1.26129482  0.92474857 -0.90725602]
---------------------------------------------
Cost at Step 205: -0.9999831444629906
Weights at Step 205: [ 0.47584426  0.18719266  1.33154192  1.79202379 -0.04462708 -0.74822003]
Bias at Step 205: [-1.10977785 -1.265427    0.92453063 -0.92310712]
---------------------------------------------
Cost at Step 210: -0.9999790517608406
Weights at Step 210: [ 0.48584547  0.17597834  1.33028455  1.79378517 -0.03978138 -0.75852048]
Bias at Step 210: [-1.13101187 -1.26898531  0.92482346 -0.93780681]
---------------------------------------------
Cost at Step 215: -0.999984711195014
Weights at Step 215: [ 0.4946749   0.16562713  1.32915967  1.79528287 -0.03462316 -0.76851533]
Bias at Step 215: [-1.15132516 -1.27221485  0.92584491 -0.9503349 ]
---------------------------------------------
Cost at Step 220: -0.9999855424488904
Weights at Step 220: [ 0.50303178  0.15574802  1.32886423  1.79677025 -0.03058318 -0.77711746]
Bias at Step 220: [-1.16955409 -1.27502675  0.92729975 -0.96212831]
---------------------------------------------
Cost at Step 225: -0.9999863772257214
Weights at Step 225: [ 0.51097377  0.14603986  1.32919454  1.79836519 -0.02733105 -0.78463519]
Bias at Step 225: [-1.18628365 -1.27783948  0.92913454 -0.97340009]
---------------------------------------------
Cost at Step 230: -0.9999898484097555
Weights at Step 230: [ 0.51836647  0.13626055  1.32967368  1.80000495 -0.02394918 -0.79187192]
Bias at Step 230: [-1.20276401 -1.28103879  0.93145415 -0.98384786]
---------------------------------------------
Cost at Step 235: -0.9999890657624451
Weights at Step 235: [ 0.52564729  0.12600301  1.32958604  1.80138672 -0.02082854 -0.79856878]
Bias at Step 235: [-1.2195148  -1.28431275  0.93422156 -0.9943289 ]
---------------------------------------------
Cost at Step 240: -0.9999882976226051
Weights at Step 240: [ 0.53229253  0.11610906  1.32908336  1.80229993 -0.01749836 -0.80503539]
Bias at Step 240: [-1.23599684 -1.28731013  0.93747248 -1.00346756]
---------------------------------------------
Cost at Step 245: -0.9999882440932369
Weights at Step 245: [ 0.53847247  0.10723924  1.33070351  1.80329887 -0.01467023 -0.81108244]
Bias at Step 245: [-1.25069646 -1.28978222  0.94081211 -1.01123413]
---------------------------------------------
Cost at Step 250: -0.9999868214455342
Weights at Step 250: [ 0.54460039  0.0980518   1.33056073  1.80410968 -0.01354454 -0.81496581]
Bias at Step 250: [-1.26394352 -1.29235223  0.94434983 -1.02022419]
---------------------------------------------
Cost at Step 255: -0.9999884214814982
Weights at Step 255: [ 0.54943379  0.08996909  1.33186446  1.8051759  -0.01056943 -0.82027142]
Bias at Step 255: [-1.27708148 -1.29545081  0.94840617 -1.02592635]
---------------------------------------------
Cost at Step 260: -0.9999905893984565
Weights at Step 260: [ 0.55392271  0.08211047  1.33002746  1.8051647  -0.00912692 -0.82330494]
Bias at Step 260: [-1.28877696 -1.29778044  0.9520992  -1.03219995]
---------------------------------------------
Cost at Step 265: -0.9999894660037792
Weights at Step 265: [ 0.55772452  0.07551996  1.33104998  1.80525639 -0.00672651 -0.82745976]
Bias at Step 265: [-1.29986939 -1.29978116  0.95585094 -1.03589225]
---------------------------------------------
Cost at Step 270: -0.9999873354125403
Weights at Step 270: [ 0.56170933  0.06903136  1.33175345  1.80537894 -0.00624539 -0.82966868]
Bias at Step 270: [-1.30907139 -1.30160912  0.9593855  -1.04092885]
---------------------------------------------
Cost at Step 275: -0.9999910126657187
Weights at Step 275: [ 0.56476346  0.06288961  1.33121415  1.80555827 -0.00483336 -0.83207027]
Bias at Step 275: [-1.3181107  -1.30426179  0.96340533 -1.04444787]
---------------------------------------------
Cost at Step 280: -0.9999911807478905
Weights at Step 280: [ 0.56790545  0.05710628  1.33293783  1.80572737 -0.00337721 -0.83486294]
Bias at Step 280: [-1.32665878 -1.30655911  0.96725682 -1.04712595]
---------------------------------------------
Cost at Step 285: -0.9999870533227304
Weights at Step 285: [ 0.57142315  0.05115447  1.33329561  1.80547158 -0.00412561 -0.83538511]
Bias at Step 285: [-1.33391608 -1.30836527  0.97068764 -1.05192006]
---------------------------------------------
Cost at Step 290: -0.9999918533581702
Weights at Step 290: [ 0.57314839  0.04641148  1.33277082  1.80515273 -0.00185471 -0.83794182]
Bias at Step 290: [-1.3420092  -1.3110848   0.97494496 -1.05274072]
---------------------------------------------
Cost at Step 295: -0.9999907581161834
Weights at Step 295: [ 5.75810409e-01  4.17373794e-02  1.33385607e+00  1.80469466e+00
 -1.37560830e-03 -8.39439431e-01]
Bias at Step 295: [-1.34861394 -1.31269908  0.97854738 -1.05489467]
---------------------------------------------
Target parameters     Learned parameters
Weights
-----------------------------------------
0.56                |  0.5782895244479023
1.24                |  1.3350283296762835
1.67                |  1.8044804399858487
-0.79               |  -0.839549739503953
Bias
-----------------------------------------
-1.44               | -1.3529586931946362
-1.43               | -1.3138918025602149
1.18                |  0.9811173767093422
-0.93               | -1.0579331212003387
 </code>
 </pre>
 </details>

---

## 9. tutorial_pasqal.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_pasqal.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost=0.0001672286566463592
Step 5: cost=0.9979047620889769
Step 10: cost=0.6109142342375409
Step 15: cost=0.9989467692733883
Step 20: cost=0.006048046345186867
Step 25: cost=0.8941419709966564
Step 30: cost=0.6746950251504293
Step 35: cost=7.001036075480078e-07
Step 40: cost=0.6766725857506097
Step 45: cost=0.3557129296721806
Step 50: cost=0.02749132423642614
Step 55: cost=0.09109423901502911
Step 60: cost=0.3024013456684429
Step 65: cost=0.01987428630678778
Step 70: cost=0.007314119488719198
Step 75: cost=0.0005591169242113734
Step 80: cost=0.00048827164327969966
Step 85: cost=6.396804707814799e-07
Step 90: cost=5.587668130241363e-05
Step 95: cost=7.117522822325351e-07
Final cost value: 1.230815538836673e-05
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_pasqal.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost=0.00016714805870154947
Step 5: cost=0.996033675363325
Step 10: cost=0.6155194682134244
Step 15: cost=0.999094333759448
Step 20: cost=0.005043049850429249
Step 25: cost=0.8981007649191639
Step 30: cost=0.6573246599021019
Step 35: cost=8.465054293083085e-07
Step 40: cost=0.6788142780522586
Step 45: cost=0.3556685123234262
Step 50: cost=0.026910206671360015
Step 55: cost=0.08898491577262835
Step 60: cost=0.31026489878494545
Step 65: cost=0.02024610375919167
Step 70: cost=0.007934105929226831
Step 75: cost=0.0005895204131158849
Step 80: cost=0.0005427646474345238
Step 85: cost=8.379720526363599e-07
Step 90: cost=5.347187868043335e-05
Step 95: cost=8.521633398927975e-07
Final cost value: 1.044140829353779e-05
 </code>
 </pre>
 </details>

---

## 10. tutorial_quanvolution.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   16384/11490434 [..............................] - ETA: 0s
 4612096/11490434 [===========>..................] - ETA: 0s
 8396800/11490434 [====================>.........] - ETA: 0s

11501568/11490434 [==============================] - 0s 0us/step
Quantum pre-processing of train images:
1/50
2/50
3/50
4/50
5/50
6/50
7/50
8/50
9/50
10/50
11/50
12/50
13/50
14/50
15/50
16/50
17/50
18/50
19/50
20/50
21/50
22/50
23/50
24/50
25/50
26/50
27/50
28/50
29/50
30/50
31/50
32/50
33/50
34/50
35/50
36/50
37/50
38/50
39/50
40/50
41/50
42/50
43/50
44/50
45/50
46/50
47/50
48/50
49/50
50/50
Quantum pre-processing of test images:
1/30
2/30
3/30
4/30
5/30
6/30
7/30
8/30
9/30
10/30
11/30
12/30
13/30
14/30
15/30
16/30
17/30
18/30
19/30
20/30
21/30
22/30
23/30
24/30
25/30
26/30
27/30
28/30
29/30
30/30
Epoch 1/30
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
    8192/11490434 [..............................] - ETA: 0s
  991232/11490434 [=>............................] - ETA: 0s
10444800/11490434 [==========================>...] - ETA: 0s
Quantum pre-processing of train images:
1/50
2/50
3/50
4/50
5/50
6/50
7/50
8/50
9/50
10/50
11/50
12/50
13/50
14/50
15/50
16/50
17/50
18/50
19/50
20/50
21/50
22/50
23/50
24/50
25/50
26/50
27/50
28/50
29/50
30/50
31/50
32/50
33/50
34/50
35/50
36/50
37/50
38/50
39/50
40/50
41/50
42/50
43/50
44/50
45/50
46/50
47/50
48/50
49/50
50/50
Quantum pre-processing of test images:
1/30
2/30
3/30
4/30
5/30
6/30
7/30
8/30
9/30
10/30
11/30
12/30
13/30
14/30
15/30
16/30
17/30
18/30
19/30
20/30
21/30
22/30
23/30
24/30
25/30
26/30
27/30
28/30
29/30
30/30
Epoch 1/30
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000
Epoch 30/30
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000
 </code>
 </pre>
 </details>

---

## 11. tutorial_ensemble_multi_qpu.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.824
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0
Choices counts: Counter({0: 110, 1: 40})
Counter({0: 55, 2: 55})
Counter({1: 37, 0: 3})
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.832
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0
Choices counts: Counter({0: 109, 1: 41})
Counter({0: 55, 2: 54})
Counter({1: 38, 0: 3})
```

---

## 12. tutorial_multiclass_classification.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_multiclass_classification.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Iter:     3 | Cost: 0.2943569 | Acc train: 0.3214286 | Acc test: 0.3684211
Iter:     6 | Cost: 0.2718905 | Acc train: 0.4910714 | Acc test: 0.5789474
Iter:     7 | Cost: 0.2201054 | Acc train: 0.4821429 | Acc test: 0.4473684
Iter:    10 | Cost: 0.2361711 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    11 | Cost: 0.2656709 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    12 | Cost: 0.1090594 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    13 | Cost: 0.0562116 | Acc train: 0.6875000 | Acc test: 0.6315789
Iter:    15 | Cost: 0.1158818 | Acc train: 0.9017857 | Acc test: 0.9473684
Iter:    17 | Cost: 0.1172837 | Acc train: 0.7589286 | Acc test: 0.7894737
Iter:    18 | Cost: 0.1232262 | Acc train: 0.7589286 | Acc test: 0.7631579
Iter:    20 | Cost: 0.1289215 | Acc train: 0.7142857 | Acc test: 0.7631579
Iter:    23 | Cost: 0.0755417 | Acc train: 0.7321429 | Acc test: 0.7631579
Iter:    24 | Cost: 0.0724914 | Acc train: 0.6964286 | Acc test: 0.7105263
Iter:    25 | Cost: 0.0919957 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    26 | Cost: 0.1054715 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    30 | Cost: 0.0872696 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    31 | Cost: 0.1019798 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    32 | Cost: 0.0757497 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    33 | Cost: 0.1152469 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    34 | Cost: 0.1455488 | Acc train: 0.7142857 | Acc test: 0.7105263
Iter:    37 | Cost: 0.0849844 | Acc train: 0.9107143 | Acc test: 0.9473684
Iter:    43 | Cost: 0.0590818 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    44 | Cost: 0.0429792 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    45 | Cost: 0.1355981 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    46 | Cost: 0.0787131 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    49 | Cost: 0.1169678 | Acc train: 0.6875000 | Acc test: 0.6578947
Iter:    50 | Cost: 0.0723402 | Acc train: 0.7142857 | Acc test: 0.7105263
Iter:    53 | Cost: 0.0851141 | Acc train: 0.9285714 | Acc test: 0.9473684
Iter:    57 | Cost: 0.0801481 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    58 | Cost: 0.1502000 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    59 | Cost: 0.0810742 | Acc train: 0.6785714 | Acc test: 0.7105263
Iter:    60 | Cost: 0.1178779 | Acc train: 0.7142857 | Acc test: 0.7105263
Iter:    61 | Cost: 0.0912379 | Acc train: 0.8035714 | Acc test: 0.8157895
Iter:    64 | Cost: 0.1046236 | Acc train: 0.8571429 | Acc test: 0.8947368
Iter:    66 | Cost: 0.0852931 | Acc train: 0.7232143 | Acc test: 0.8157895
Iter:    68 | Cost: 0.0670566 | Acc train: 0.6964286 | Acc test: 0.6842105
Iter:    72 | Cost: 0.0925488 | Acc train: 0.6964286 | Acc test: 0.6578947
Iter:    73 | Cost: 0.0831474 | Acc train: 0.7232143 | Acc test: 0.7368421
Iter:    75 | Cost: 0.0768720 | Acc train: 0.8482143 | Acc test: 0.8947368
Iter:    77 | Cost: 0.0461329 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    78 | Cost: 0.0674305 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    79 | Cost: 0.0276814 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    80 | Cost: 0.0586606 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    82 | Cost: 0.0479530 | Acc train: 0.9375000 | Acc test: 0.9473684
Iter:    83 | Cost: 0.0823876 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    86 | Cost: 0.0964098 | Acc train: 0.9553571 | Acc test: 0.9736842
Iter:    87 | Cost: 0.0150624 | Acc train: 0.9375000 | Acc test: 0.9210526
Iter:    91 | Cost: 0.0351690 | Acc train: 0.9107143 | Acc test: 0.9210526
Iter:    92 | Cost: 0.0555153 | Acc train: 0.9017857 | Acc test: 0.9210526
Iter:    93 | Cost: 0.0339653 | Acc train: 0.8750000 | Acc test: 0.8684211
Iter:    95 | Cost: 0.0358331 | Acc train: 0.8482143 | Acc test: 0.8947368
Iter:    97 | Cost: 0.0946532 | Acc train: 0.8035714 | Acc test: 0.8947368
Iter:    98 | Cost: 0.0701062 | Acc train: 0.8839286 | Acc test: 0.8684211
Iter:    99 | Cost: 0.0827177 | Acc train: 0.9642857 | Acc test: 0.9473684
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_multiclass_classification.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Iter:     3 | Cost: 0.2943570 | Acc train: 0.3214286 | Acc test: 0.3684211
Iter:     6 | Cost: 0.2718902 | Acc train: 0.4910714 | Acc test: 0.5789474
Iter:     7 | Cost: 0.2201053 | Acc train: 0.4821429 | Acc test: 0.4473684
Iter:    10 | Cost: 0.2361710 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    11 | Cost: 0.2656707 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    12 | Cost: 0.1090596 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    13 | Cost: 0.0562117 | Acc train: 0.6875000 | Acc test: 0.6315789
Iter:    15 | Cost: 0.1158819 | Acc train: 0.9017857 | Acc test: 0.9473684
Iter:    17 | Cost: 0.1172836 | Acc train: 0.7589286 | Acc test: 0.7894737
Iter:    18 | Cost: 0.1232261 | Acc train: 0.7589286 | Acc test: 0.7631579
Iter:    20 | Cost: 0.1289214 | Acc train: 0.7142857 | Acc test: 0.7631579
Iter:    23 | Cost: 0.0755416 | Acc train: 0.7321429 | Acc test: 0.7631579
Iter:    24 | Cost: 0.0724915 | Acc train: 0.6964286 | Acc test: 0.7105263
Iter:    25 | Cost: 0.0919956 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    26 | Cost: 0.1054716 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    30 | Cost: 0.0872695 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    31 | Cost: 0.1019797 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    32 | Cost: 0.0757496 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    33 | Cost: 0.1152470 | Acc train: 0.6607143 | Acc test: 0.6842105
Iter:    34 | Cost: 0.1455489 | Acc train: 0.7142857 | Acc test: 0.7105263
Iter:    37 | Cost: 0.0849843 | Acc train: 0.9107143 | Acc test: 0.9473684
Iter:    43 | Cost: 0.0590819 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    44 | Cost: 0.0429791 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    45 | Cost: 0.1355980 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    46 | Cost: 0.0787130 | Acc train: 0.6785714 | Acc test: 0.6315789
Iter:    49 | Cost: 0.1169677 | Acc train: 0.6875000 | Acc test: 0.6578947
Iter:    50 | Cost: 0.0723401 | Acc train: 0.7142857 | Acc test: 0.7105263
Iter:    53 | Cost: 0.0851140 | Acc train: 0.9285714 | Acc test: 0.9473684
Iter:    57 | Cost: 0.0801482 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    58 | Cost: 0.1502001 | Acc train: 0.6785714 | Acc test: 0.6842105
Iter:    59 | Cost: 0.0810743 | Acc train: 0.6785714 | Acc test: 0.7105263
Iter:    60 | Cost: 0.1178781 | Acc train: 0.7142857 | Acc test: 0.7105263
Iter:    61 | Cost: 0.0912380 | Acc train: 0.8035714 | Acc test: 0.8157895
Iter:    64 | Cost: 0.1046238 | Acc train: 0.8571429 | Acc test: 0.8947368
Iter:    66 | Cost: 0.0852930 | Acc train: 0.7232143 | Acc test: 0.8157895
Iter:    68 | Cost: 0.0670565 | Acc train: 0.6964286 | Acc test: 0.6842105
Iter:    72 | Cost: 0.0925487 | Acc train: 0.6964286 | Acc test: 0.6578947
Iter:    73 | Cost: 0.0831472 | Acc train: 0.7232143 | Acc test: 0.7368421
Iter:    75 | Cost: 0.0768719 | Acc train: 0.8482143 | Acc test: 0.8947368
Iter:    77 | Cost: 0.0461330 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    78 | Cost: 0.0674306 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    79 | Cost: 0.0276815 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    80 | Cost: 0.0586605 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    82 | Cost: 0.0479531 | Acc train: 0.9375000 | Acc test: 0.9473684
Iter:    83 | Cost: 0.0823874 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    86 | Cost: 0.0964097 | Acc train: 0.9553571 | Acc test: 0.9736842
Iter:    87 | Cost: 0.0150623 | Acc train: 0.9375000 | Acc test: 0.9210526
Iter:    91 | Cost: 0.0351689 | Acc train: 0.9107143 | Acc test: 0.9210526
Iter:    92 | Cost: 0.0555152 | Acc train: 0.9017857 | Acc test: 0.9210526
Iter:    93 | Cost: 0.0339652 | Acc train: 0.8750000 | Acc test: 0.8684211
Iter:    95 | Cost: 0.0358330 | Acc train: 0.8482143 | Acc test: 0.8947368
Iter:    97 | Cost: 0.0946534 | Acc train: 0.8035714 | Acc test: 0.8947368
Iter:    98 | Cost: 0.0701063 | Acc train: 0.8839286 | Acc test: 0.8684211
Iter:    99 | Cost: 0.0827179 | Acc train: 0.9642857 | Acc test: 0.9473684
 </code>
 </pre>
 </details>

---

## 13. tutorial_adaptive_circuits.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157616985
Excitation : [0, 1, 2, 5], Gradient: -1.0842021724855047e-19
Excitation : [0, 1, 2, 7], Gradient: 1.0842021724855052e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170162428
Excitation : [0, 1, 3, 4], Gradient: -2.71050543121376e-20
Excitation : [0, 1, 3, 6], Gradient: -2.710505431213759e-20
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170162448
Excitation : [0, 1, 4, 5], Gradient: -0.02358152902066405
Excitation : [0, 1, 5, 8], Gradient: -1.8973538018496308e-19
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020664065
Excitation : [0, 1, 7, 8], Gradient: 5.421010862427539e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485601669
Excitation : [0, 2], Gradient: -0.005062536239388248
Excitation : [0, 4], Gradient: -4.045427276296455e-18
Excitation : [0, 6], Gradient: 2.1802350921966793e-18
Excitation : [0, 8], Gradient: -0.0009448044625848142
Excitation : [1, 3], Gradient: 0.004926616877055942
Excitation : [1, 5], Gradient: -4.5878327696512595e-18
Excitation : [1, 7], Gradient: -1.7686993299857877e-18
Excitation : [1, 9], Gradient: 0.0014535534854159644
[[0, 2], [0, 8], [1, 3], [1, 9]]
n = 0,  E = -7.86266587 H, t = 2.03 s
n = 1,  E = -7.87094621 H, t = 2.54 s
n = 2,  E = -7.87563100 H, t = 2.06 s
n = 3,  E = -7.87829146 H, t = 2.57 s
n = 4,  E = -7.87981705 H, t = 2.00 s
n = 5,  E = -7.88070477 H, t = 2.55 s
n = 6,  E = -7.88123143 H, t = 2.02 s
n = 7,  E = -7.88155161 H, t = 2.53 s
n = 8,  E = -7.88175217 H, t = 2.01 s
n = 9,  E = -7.88188237 H, t = 2.53 s
n = 10,  E = -7.88197041 H, t = 2.00 s
n = 11,  E = -7.88203267 H, t = 2.00 s
n = 12,  E = -7.88207879 H, t = 2.48 s
n = 13,  E = -7.88211452 H, t = 2.52 s
n = 14,  E = -7.88214335 H, t = 2.04 s
n = 15,  E = -7.88216743 H, t = 2.52 s
n = 16,  E = -7.88218814 H, t = 2.11 s
n = 17,  E = -7.88220634 H, t = 2.03 s
n = 18,  E = -7.88222261 H, t = 2.51 s
n = 19,  E = -7.88223734 H, t = 2.54 s
<1024x1024 sparse matrix of type '<class 'numpy.complex128'>'
    with 11264 stored elements in COOrdinate format>
n = 0,  E = -7.86266587 H, t = 0.12 s
n = 1,  E = -7.87094621 H, t = 0.12 s
n = 2,  E = -7.87563100 H, t = 0.12 s
n = 3,  E = -7.87829146 H, t = 0.12 s
n = 4,  E = -7.87981705 H, t = 0.12 s
n = 5,  E = -7.88070477 H, t = 0.12 s
n = 6,  E = -7.88123143 H, t = 0.13 s
n = 7,  E = -7.88155161 H, t = 0.13 s
n = 8,  E = -7.88175217 H, t = 0.12 s
n = 9,  E = -7.88188237 H, t = 0.13 s
n = 10,  E = -7.88197041 H, t = 0.13 s
n = 11,  E = -7.88203267 H, t = 0.12 s
n = 12,  E = -7.88207879 H, t = 0.12 s
n = 13,  E = -7.88211452 H, t = 0.12 s
n = 14,  E = -7.88214335 H, t = 0.12 s
n = 15,  E = -7.88216743 H, t = 0.12 s
n = 16,  E = -7.88218814 H, t = 0.12 s
n = 17,  E = -7.88220634 H, t = 0.12 s
n = 18,  E = -7.88222261 H, t = 0.13 s
n = 19,  E = -7.88223734 H, t = 0.13 s
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157624169
Excitation : [0, 1, 2, 5], Gradient: 3.3881317890172014e-20
Excitation : [0, 1, 2, 7], Gradient: 3.3881317890172014e-20
Excitation : [0, 1, 2, 9], Gradient: 0.034264511701633305
Excitation : [0, 1, 3, 4], Gradient: 2.032879073410333e-20
Excitation : [0, 1, 3, 6], Gradient: -2.032879073410333e-20
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170163325
Excitation : [0, 1, 4, 5], Gradient: -0.023581529020665994
Excitation : [0, 1, 5, 8], Gradient: -3.3881317890171514e-20
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020665997
Excitation : [0, 1, 7, 8], Gradient: -3.3881317890171514e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485601202
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Excitation : [0, 2], Gradient: -0.005062536239379516
Excitation : [0, 4], Gradient: 4.75975558441247e-18
Excitation : [0, 6], Gradient: -8.980041231095072e-19
Excitation : [0, 8], Gradient: -0.0009448044625839003
Excitation : [1, 3], Gradient: 0.004926616877047353
Excitation : [1, 5], Gradient: 1.3160755791966491e-18
Excitation : [1, 7], Gradient: 1.7726247697948774e-18
Excitation : [1, 9], Gradient: 0.0014535534854143214
[[0, 2], [0, 8], [1, 3], [1, 9]]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
n = 0,  E = -7.86266587 H, t = 2.08 s
n = 1,  E = -7.87094621 H, t = 2.56 s
n = 2,  E = -7.87563100 H, t = 2.60 s
n = 3,  E = -7.87829146 H, t = 2.09 s
n = 4,  E = -7.87981705 H, t = 2.57 s
n = 5,  E = -7.88070477 H, t = 2.60 s
n = 6,  E = -7.88123143 H, t = 2.09 s
n = 7,  E = -7.88155161 H, t = 2.56 s
n = 8,  E = -7.88175217 H, t = 2.59 s
n = 9,  E = -7.88188237 H, t = 2.08 s
n = 10,  E = -7.88197041 H, t = 2.56 s
n = 11,  E = -7.88203267 H, t = 2.60 s
n = 12,  E = -7.88207879 H, t = 2.09 s
n = 13,  E = -7.88211452 H, t = 2.56 s
n = 14,  E = -7.88214335 H, t = 2.60 s
n = 15,  E = -7.88216743 H, t = 2.09 s
n = 16,  E = -7.88218814 H, t = 2.56 s
n = 17,  E = -7.88220634 H, t = 2.60 s
n = 18,  E = -7.88222261 H, t = 2.09 s
n = 19,  E = -7.88223734 H, t = 2.56 s
<1024x1024 sparse matrix of type '<class 'numpy.complex128'>'
    with 11264 stored elements in COOrdinate format>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
n = 0,  E = -7.86266587 H, t = 0.11 s
n = 1,  E = -7.87094621 H, t = 0.11 s
n = 2,  E = -7.87563100 H, t = 0.11 s
n = 3,  E = -7.87829146 H, t = 0.11 s
n = 4,  E = -7.87981705 H, t = 0.11 s
n = 5,  E = -7.88070477 H, t = 0.11 s
n = 6,  E = -7.88123143 H, t = 0.11 s
n = 7,  E = -7.88155161 H, t = 0.11 s
n = 8,  E = -7.88175217 H, t = 0.11 s
n = 9,  E = -7.88188237 H, t = 0.11 s
n = 10,  E = -7.88197041 H, t = 0.11 s
n = 11,  E = -7.88203267 H, t = 0.11 s
n = 12,  E = -7.88207879 H, t = 0.11 s
n = 13,  E = -7.88211452 H, t = 0.11 s
n = 14,  E = -7.88214335 H, t = 0.11 s
n = 15,  E = -7.88216743 H, t = 0.11 s
n = 16,  E = -7.88218814 H, t = 0.11 s
 </code>
 </pre>
 </details>

---

## 14. tutorial_data_reuploading_classifier.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier.html):

```
Layer 2: [-2.29400846 -1.18534645  0.32099705]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_data_reuploading_classifier.html):

```
Layer 2: [-2.29400846 -1.18534645  0.32099704]
```

---

## 15. tutorial_classical_shadows.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_classical_shadows.html):

```
(0.16156422871415568+9.967876155406687e-20j)
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_classical_shadows.html):

```
(0.16156422871415568+0j)
```

---

## 16. tutorial_vqe_spin_sectors.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_spin_sectors.html):

```
Step = 0, Energy = -0.09929556 Ha, S = 0.1014
Step = 4, Energy = -0.87153518 Ha, S = 0.0982
Step = 8, Energy = -1.11692841 Ha, S = 0.0087
Step = 12, Energy = -1.13529755 Ha, S = 0.0004
Step = 16, Energy = -1.13614887 Ha, S = 0.0000
Step = 20, Energy = -1.13618734 Ha, S = 0.0000
Final value of the ground-state energy = -1.13618832 Ha
Optimal value of the circuit parameters = [3.14350662 3.14087516 2.93185887]
[[1, 2]]
[]
Step = 0, Energy = 0.31463320 Ha, S = 0.3539
Step = 4, Energy = -0.38517129 Ha, S = 0.9391
Step = 8, Energy = -0.47698617 Ha, S = 0.9991
Step = 12, Energy = -0.47842742 Ha, S = 1.0000
Step = 16, Energy = -0.47844666 Ha, S = 1.0000
Final value of the energy = -0.47844666 Ha
Optimal value of the circuit parameters = [3.14259046]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_spin_sectors.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step = 0, Energy = -0.09929556 Ha, S = 0.1014
Step = 4, Energy = -0.87153518 Ha, S = 0.0982
Step = 8, Energy = -1.11692841 Ha, S = 0.0087
Step = 12, Energy = -1.13529755 Ha, S = 0.0004
Step = 16, Energy = -1.13614887 Ha, S = 0.0000
Step = 20, Energy = -1.13618734 Ha, S = 0.0000
Final value of the ground-state energy = -1.13618832 Ha
Optimal value of the circuit parameters = [3.14350662 3.14087516 2.93185887]
[[1, 2]]
[]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step = 0, Energy = 0.31463320 Ha, S = 0.3539
Step = 4, Energy = -0.38517129 Ha, S = 0.9391
Step = 8, Energy = -0.47698617 Ha, S = 0.9991
Step = 12, Energy = -0.47842742 Ha, S = 1.0000
Step = 16, Energy = -0.47844666 Ha, S = 1.0000
```

---

## 17. tutorial_vqe_parallel.html <a name="demo16"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.90
Evaluation time: 306.69 s
Evaluation time: 105.73 s
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 3.08
Evaluation time: 261.58 s
Evaluation time: 84.81 s
```

---

## 18. tutorial_qnn_module_tf.html <a name="demo17"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 14s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 15s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 14s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 14s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 14s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 14s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 29s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 29s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 29s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 29s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 29s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 29s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 7s - loss: 0.3931 - accuracy: 0.7067 - val_loss: 0.2683 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.2107 - accuracy: 0.8600 - val_loss: 0.1992 - val_accuracy: 0.8200
30/30 - 7s - loss: 0.1670 - accuracy: 0.8800 - val_loss: 0.1854 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.1602 - accuracy: 0.8800 - val_loss: 0.1732 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.1514 - accuracy: 0.8800 - val_loss: 0.1692 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.1433 - accuracy: 0.8800 - val_loss: 0.1787 - val_accuracy: 0.8200
30/30 - 15s - loss: 0.4068 - accuracy: 0.6600 - val_loss: 0.3008 - val_accuracy: 0.7400
30/30 - 14s - loss: 0.2845 - accuracy: 0.7733 - val_loss: 0.2298 - val_accuracy: 0.8200
30/30 - 15s - loss: 0.2180 - accuracy: 0.8067 - val_loss: 0.1976 - val_accuracy: 0.8200
30/30 - 15s - loss: 0.1904 - accuracy: 0.8533 - val_loss: 0.1809 - val_accuracy: 0.8200
30/30 - 14s - loss: 0.1702 - accuracy: 0.8600 - val_loss: 0.1719 - val_accuracy: 0.8600
30/30 - 14s - loss: 0.1538 - accuracy: 0.8600 - val_loss: 0.1862 - val_accuracy: 0.8400
```

---

## 19. tutorial_backprop.html <a name="demo18"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
Forward pass (best of 3): 0.007647444400026871 sec per loop
Gradient computation (best of 3): 2.889219030200002 sec per loop
2.7530799840096734
0.9358535378025419
Forward pass (best of 3): 0.0577198712999234 sec per loop
Backward pass (best of 3): 0.10437145799996869 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
Forward pass (best of 3): 0.011453844200059394 sec per loop
Gradient computation (best of 3): 4.242158207600005 sec per loop
4.1233839120213815
0.9358535378025427
Forward pass (best of 3): 0.048751778400037436 sec per loop
Backward pass (best of 3): 0.1061622050000551 sec per loop
```

---

## 20. tutorial_state_preparation.html <a name="demo19"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_state_preparation.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/torch/autograd/__init__.py:156: UserWarning: Casting complex values to real discards the imaginary part (Triggered internally at  ../aten/src/ATen/native/Copy.cpp:244.)
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_state_preparation.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/torch/autograd/__init__.py:149: UserWarning: Casting complex values to real discards the imaginary part (Triggered internally at  /pytorch/aten/src/ATen/native/Copy.cpp:240.)
```

---

## 21. tutorial_general_parshift.html <a name="demo20"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_general_parshift.html):

```
For 2 qubits the spectrum is [-2.0, -1.0, 0.0, 1.0, 2.0].
For 4 qubits the spectrum is [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0].
For 5 qubits the spectrum is [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0].
Second-order finite difference:    [ 0.26814   1.696854 -2.055918 -7.236953]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_general_parshift.html):

```
For 2 qubits the spectrum is [-2.0, -1.0, 0, 1.0, 2.0].
For 4 qubits the spectrum is [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0].
For 5 qubits the spectrum is [-5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0].
Second-order finite difference:    [ 0.26814   1.696853 -2.055918 -7.236953]
```

---

## 22. tutorial_doubly_stochastic.html <a name="demo21"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_doubly_stochastic.html):

```
Stochastic gradient descent (shots=100) min energy =  -4.60065517691614
Adaptive QSGD min energy =  -4.592548741613157
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_doubly_stochastic.html):

```
Stochastic gradient descent (shots=100) min energy =  -4.600655176916144
Adaptive QSGD min energy =  -4.592548741613161
```

---

## 23. tutorial_variational_classifier.html <a name="demo22"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Iter:     1 | Cost: 3.4355534 | Accuracy: 0.5000000
Iter:     2 | Cost: 1.9287800 | Accuracy: 0.5000000
Iter:     3 | Cost: 2.0341238 | Accuracy: 0.5000000
Iter:     4 | Cost: 1.6372574 | Accuracy: 0.5000000
Iter:     5 | Cost: 1.3025395 | Accuracy: 0.6250000
Iter:     6 | Cost: 1.4555019 | Accuracy: 0.3750000
Iter:     7 | Cost: 1.4492786 | Accuracy: 0.5000000
Iter:     8 | Cost: 0.6510286 | Accuracy: 0.8750000
Iter:     9 | Cost: 0.0566074 | Accuracy: 1.0000000
Iter:    10 | Cost: 0.0053045 | Accuracy: 1.0000000
Iter:    11 | Cost: 0.0809483 | Accuracy: 1.0000000
Iter:    12 | Cost: 0.1115426 | Accuracy: 1.0000000
Iter:    13 | Cost: 0.1460257 | Accuracy: 1.0000000
Iter:    14 | Cost: 0.0877037 | Accuracy: 1.0000000
Iter:    15 | Cost: 0.0361311 | Accuracy: 1.0000000
Iter:    16 | Cost: 0.0040937 | Accuracy: 1.0000000
Iter:    17 | Cost: 0.0004899 | Accuracy: 1.0000000
Iter:    18 | Cost: 0.0005290 | Accuracy: 1.0000000
Iter:    19 | Cost: 0.0024304 | Accuracy: 1.0000000
Iter:    20 | Cost: 0.0062137 | Accuracy: 1.0000000
Iter:    21 | Cost: 0.0088864 | Accuracy: 1.0000000
Iter:    22 | Cost: 0.0201912 | Accuracy: 1.0000000
Iter:    23 | Cost: 0.0060335 | Accuracy: 1.0000000
Iter:    24 | Cost: 0.0036153 | Accuracy: 1.0000000
Iter:    25 | Cost: 0.0012741 | Accuracy: 1.0000000
x               :  [0.53896774 0.79503606 0.27826503 0.        ]
angles          :  [ 0.56397465 -0.          0.         -0.97504604  0.97504604]
amplitude vector:  [ 5.38967743e-01  7.95036065e-01  2.78265032e-01 -2.77555756e-17]
First X sample (original)  : [0.4  0.75]
First X sample (padded)    : [0.4  0.75 0.3  0.  ]
First X sample (normalized): [0.44376016 0.83205029 0.33282012 0.        ]
First features sample      : [ 0.67858523 -0.          0.         -1.080839    1.080839  ]
Iter:     1 | Cost: 1.4490948 | Acc train: 0.4933333 | Acc validation: 0.5600000
Iter:     2 | Cost: 1.3309953 | Acc train: 0.4933333 | Acc validation: 0.5600000
Iter:     3 | Cost: 1.1582178 | Acc train: 0.4533333 | Acc validation: 0.5600000
Iter:     4 | Cost: 0.9795035 | Acc train: 0.4800000 | Acc validation: 0.5600000
Iter:     5 | Cost: 0.8857893 | Acc train: 0.6400000 | Acc validation: 0.7600000
Iter:     6 | Cost: 0.8587935 | Acc train: 0.7066667 | Acc validation: 0.7600000
Iter:     7 | Cost: 0.8496204 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:     8 | Cost: 0.8200972 | Acc train: 0.7333333 | Acc validation: 0.6800000
Iter:     9 | Cost: 0.8027511 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    10 | Cost: 0.7695152 | Acc train: 0.8000000 | Acc validation: 0.7600000
Iter:    11 | Cost: 0.7437432 | Acc train: 0.8133333 | Acc validation: 0.9600000
Iter:    12 | Cost: 0.7569196 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    13 | Cost: 0.7887487 | Acc train: 0.6533333 | Acc validation: 0.7200000
Iter:    14 | Cost: 0.8401458 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    15 | Cost: 0.8651830 | Acc train: 0.5600000 | Acc validation: 0.6000000
Iter:    16 | Cost: 0.8726113 | Acc train: 0.5600000 | Acc validation: 0.6000000
Iter:    17 | Cost: 0.8389732 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    18 | Cost: 0.8004839 | Acc train: 0.6266667 | Acc validation: 0.6400000
Iter:    19 | Cost: 0.7592044 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    20 | Cost: 0.7332872 | Acc train: 0.7733333 | Acc validation: 0.8000000
Iter:    21 | Cost: 0.7184319 | Acc train: 0.8800000 | Acc validation: 0.9600000
Iter:    22 | Cost: 0.7336631 | Acc train: 0.8133333 | Acc validation: 0.7200000
Iter:    23 | Cost: 0.7503193 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    24 | Cost: 0.7608474 | Acc train: 0.5866667 | Acc validation: 0.5200000
Iter:    25 | Cost: 0.7443533 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    26 | Cost: 0.7383224 | Acc train: 0.7066667 | Acc validation: 0.6400000
Iter:    27 | Cost: 0.7322155 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    28 | Cost: 0.7384175 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    29 | Cost: 0.7393227 | Acc train: 0.6400000 | Acc validation: 0.6400000
Iter:    30 | Cost: 0.7251903 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:    31 | Cost: 0.7125040 | Acc train: 0.7866667 | Acc validation: 0.6800000
Iter:    32 | Cost: 0.6932690 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    33 | Cost: 0.6800562 | Acc train: 0.9200000 | Acc validation: 1.0000000
Iter:    34 | Cost: 0.6763140 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    35 | Cost: 0.6790040 | Acc train: 0.8933333 | Acc validation: 0.8800000
Iter:    36 | Cost: 0.6936199 | Acc train: 0.7600000 | Acc validation: 0.7200000
Iter:    37 | Cost: 0.6767184 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    38 | Cost: 0.6712470 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    39 | Cost: 0.6747390 | Acc train: 0.7600000 | Acc validation: 0.7600000
Iter:    40 | Cost: 0.6845696 | Acc train: 0.6666667 | Acc validation: 0.6400000
Iter:    41 | Cost: 0.6703303 | Acc train: 0.7333333 | Acc validation: 0.7200000
Iter:    42 | Cost: 0.6238401 | Acc train: 0.8933333 | Acc validation: 0.8400000
Iter:    43 | Cost: 0.6028185 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    44 | Cost: 0.5936355 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    45 | Cost: 0.5722417 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    46 | Cost: 0.5617923 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    47 | Cost: 0.5413240 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    48 | Cost: 0.5239643 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    49 | Cost: 0.5100842 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    50 | Cost: 0.5006861 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    51 | Cost: 0.4821672 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    52 | Cost: 0.4579575 | Acc train: 0.9600000 | Acc validation: 1.0000000
Iter:    53 | Cost: 0.4397479 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    54 | Cost: 0.4326879 | Acc train: 0.9600000 | Acc validation: 0.9200000
Iter:    55 | Cost: 0.4351511 | Acc train: 0.9466667 | Acc validation: 0.9200000
Iter:    56 | Cost: 0.4328988 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    57 | Cost: 0.4149892 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    58 | Cost: 0.3755246 | Acc train: 0.9600000 | Acc validation: 0.9200000
Iter:    59 | Cost: 0.3468994 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    60 | Cost: 0.3297071 | Acc train: 1.0000000 | Acc validation: 1.0000000
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_variational_classifier.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Iter:     1 | Cost: 3.4355534 | Accuracy: 0.5000000
Iter:     2 | Cost: 1.9287800 | Accuracy: 0.5000000
Iter:     3 | Cost: 2.0341238 | Accuracy: 0.5000000
Iter:     4 | Cost: 1.6372574 | Accuracy: 0.5000000
Iter:     5 | Cost: 1.3025395 | Accuracy: 0.6250000
Iter:     6 | Cost: 1.4555019 | Accuracy: 0.3750000
Iter:     7 | Cost: 1.4492786 | Accuracy: 0.5000000
Iter:     8 | Cost: 0.6510286 | Accuracy: 0.8750000
Iter:     9 | Cost: 0.0566074 | Accuracy: 1.0000000
Iter:    10 | Cost: 0.0053045 | Accuracy: 1.0000000
Iter:    11 | Cost: 0.0809483 | Accuracy: 1.0000000
Iter:    12 | Cost: 0.1115426 | Accuracy: 1.0000000
Iter:    13 | Cost: 0.1460257 | Accuracy: 1.0000000
Iter:    14 | Cost: 0.0877037 | Accuracy: 1.0000000
Iter:    15 | Cost: 0.0361311 | Accuracy: 1.0000000
Iter:    16 | Cost: 0.0040937 | Accuracy: 1.0000000
Iter:    17 | Cost: 0.0004899 | Accuracy: 1.0000000
Iter:    18 | Cost: 0.0005290 | Accuracy: 1.0000000
Iter:    19 | Cost: 0.0024304 | Accuracy: 1.0000000
Iter:    20 | Cost: 0.0062137 | Accuracy: 1.0000000
Iter:    21 | Cost: 0.0088864 | Accuracy: 1.0000000
Iter:    22 | Cost: 0.0201912 | Accuracy: 1.0000000
Iter:    23 | Cost: 0.0060335 | Accuracy: 1.0000000
Iter:    24 | Cost: 0.0036153 | Accuracy: 1.0000000
Iter:    25 | Cost: 0.0012741 | Accuracy: 1.0000000
x               :  [0.53896774 0.79503606 0.27826503 0.        ]
angles          :  [ 0.56397465 -0.          0.         -0.97504604  0.97504604]
amplitude vector:  [ 5.38967743e-01  7.95036065e-01  2.78265032e-01 -2.77555756e-17]
First X sample (original)  : [0.4  0.75]
First X sample (padded)    : [0.4  0.75 0.3  0.  ]
First X sample (normalized): [0.44376016 0.83205029 0.33282012 0.        ]
First features sample      : [ 0.67858523 -0.          0.         -1.080839    1.080839  ]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Iter:     1 | Cost: 1.4490948 | Acc train: 0.4933333 | Acc validation: 0.5600000
Iter:     2 | Cost: 1.3309953 | Acc train: 0.4933333 | Acc validation: 0.5600000
Iter:     3 | Cost: 1.1582178 | Acc train: 0.4533333 | Acc validation: 0.5600000
Iter:     4 | Cost: 0.9795035 | Acc train: 0.4800000 | Acc validation: 0.5600000
Iter:     5 | Cost: 0.8857893 | Acc train: 0.6400000 | Acc validation: 0.7600000
Iter:     6 | Cost: 0.8587935 | Acc train: 0.7066667 | Acc validation: 0.7600000
Iter:     7 | Cost: 0.8496204 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:     8 | Cost: 0.8200972 | Acc train: 0.7333333 | Acc validation: 0.6800000
Iter:     9 | Cost: 0.8027511 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    10 | Cost: 0.7695152 | Acc train: 0.8000000 | Acc validation: 0.7600000
Iter:    11 | Cost: 0.7437432 | Acc train: 0.8133333 | Acc validation: 0.9600000
Iter:    12 | Cost: 0.7569196 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    13 | Cost: 0.7887487 | Acc train: 0.6533333 | Acc validation: 0.7200000
Iter:    14 | Cost: 0.8401458 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    15 | Cost: 0.8651830 | Acc train: 0.5600000 | Acc validation: 0.6000000
Iter:    16 | Cost: 0.8726113 | Acc train: 0.5600000 | Acc validation: 0.6000000
Iter:    17 | Cost: 0.8389732 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    18 | Cost: 0.8004839 | Acc train: 0.6266667 | Acc validation: 0.6400000
Iter:    19 | Cost: 0.7592044 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    20 | Cost: 0.7332872 | Acc train: 0.7733333 | Acc validation: 0.8000000
Iter:    21 | Cost: 0.7184319 | Acc train: 0.8800000 | Acc validation: 0.9600000
Iter:    22 | Cost: 0.7336631 | Acc train: 0.8133333 | Acc validation: 0.7200000
Iter:    23 | Cost: 0.7503193 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    24 | Cost: 0.7608474 | Acc train: 0.5866667 | Acc validation: 0.5200000
Iter:    25 | Cost: 0.7443533 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    26 | Cost: 0.7383224 | Acc train: 0.7066667 | Acc validation: 0.6400000
Iter:    27 | Cost: 0.7322155 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    28 | Cost: 0.7384175 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    29 | Cost: 0.7393227 | Acc train: 0.6400000 | Acc validation: 0.6400000
Iter:    30 | Cost: 0.7251903 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:    31 | Cost: 0.7125040 | Acc train: 0.7866667 | Acc validation: 0.6800000
Iter:    32 | Cost: 0.6932690 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    33 | Cost: 0.6800562 | Acc train: 0.9200000 | Acc validation: 1.0000000
Iter:    34 | Cost: 0.6763140 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    35 | Cost: 0.6790040 | Acc train: 0.8933333 | Acc validation: 0.8800000
Iter:    36 | Cost: 0.6936199 | Acc train: 0.7600000 | Acc validation: 0.7200000
Iter:    37 | Cost: 0.6767184 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    38 | Cost: 0.6712470 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    39 | Cost: 0.6747390 | Acc train: 0.7600000 | Acc validation: 0.7600000
Iter:    40 | Cost: 0.6845696 | Acc train: 0.6666667 | Acc validation: 0.6400000
Iter:    41 | Cost: 0.6703303 | Acc train: 0.7333333 | Acc validation: 0.7200000
Iter:    42 | Cost: 0.6238401 | Acc train: 0.8933333 | Acc validation: 0.8400000
Iter:    43 | Cost: 0.6028185 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    44 | Cost: 0.5936355 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    45 | Cost: 0.5722417 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    46 | Cost: 0.5617923 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    47 | Cost: 0.5413240 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    48 | Cost: 0.5239643 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    49 | Cost: 0.5100842 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    50 | Cost: 0.5006861 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    51 | Cost: 0.4821672 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    52 | Cost: 0.4579575 | Acc train: 0.9600000 | Acc validation: 1.0000000
Iter:    53 | Cost: 0.4397479 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    54 | Cost: 0.4326879 | Acc train: 0.9600000 | Acc validation: 0.9200000
Iter:    55 | Cost: 0.4351511 | Acc train: 0.9466667 | Acc validation: 0.9200000
Iter:    56 | Cost: 0.4328988 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    57 | Cost: 0.4149892 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    58 | Cost: 0.3755246 | Acc train: 0.9600000 | Acc validation: 0.9200000
 </code>
 </pre>
 </details>

---

## 24. tutorial_falqon.html <a name="demo23"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_falqon.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 1, Cost = -2.4265436197783425
Step 2, Cost = -5.45183841811118
Step 3, Cost = -5.058939064534102
Step 4, Cost = 0.666377989107735
Step 5, Cost = -3.961765919151042
Step 6, Cost = -6.012336027057502
Step 7, Cost = -6.383828240291059
Step 8, Cost = -6.568581722318154
Step 9, Cost = -6.652767426710378
Step 10, Cost = -6.718062615729133
Step 11, Cost = -6.7639477436093
Step 12, Cost = -6.8048574666097235
Step 13, Cost = -6.8394030587361705
Step 14, Cost = -6.8714592635528415
Step 15, Cost = -6.8997469754809675
Step 16, Cost = -6.925884328592705
Step 17, Cost = -6.9492295078855975
Step 18, Cost = -6.97059412505724
Step 19, Cost = -6.989907329921377
Step 20, Cost = -7.007623105822664
Step 21, Cost = -7.0239860498803415
Step 22, Cost = -7.039304856521962
Step 23, Cost = -7.053894937286077
Step 24, Cost = -7.067988454154516
Step 25, Cost = -7.081842534715257
Step 26, Cost = -7.095617260802714
Step 27, Cost = -7.109472588274413
Step 28, Cost = -7.123480825409048
Step 29, Cost = -7.137684426026884
Step 30, Cost = -7.152041022693121
Step 31, Cost = -7.166453310287251
Step 32, Cost = -7.1807483416093705
Step 33, Cost = -7.194694917926645
Step 34, Cost = -7.208028603663313
Step 35, Cost = -7.22045639587035
Step 36, Cost = -7.231727330032172
Step 37, Cost = -7.241565955502995
Step 38, Cost = -7.249767410209228
Step 39, Cost = -7.255782895664823
Step 40, Cost = -7.258987907014075
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_falqon.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step 1, Cost = -2.4265436197783448
Step 2, Cost = -5.451838418111176
Step 3, Cost = -5.05893906453409
Step 4, Cost = 0.6663779891077449
Step 5, Cost = -3.9617659191509746
Step 6, Cost = -6.012336027057521
Step 7, Cost = -6.383828240291071
Step 8, Cost = -6.5685817223181155
Step 9, Cost = -6.652767426710387
Step 10, Cost = -6.718062615729132
Step 11, Cost = -6.763947743609312
Step 12, Cost = -6.80485746660975
Step 13, Cost = -6.8394030587362
Step 14, Cost = -6.871459263552881
Step 15, Cost = -6.899746975480981
Step 16, Cost = -6.925884328592719
Step 17, Cost = -6.949229507885613
Step 18, Cost = -6.970594125057218
Step 19, Cost = -6.989907329921348
Step 20, Cost = -7.007623105822645
Step 21, Cost = -7.023986049880396
Step 22, Cost = -7.03930485652197
Step 23, Cost = -7.053894937286083
Step 24, Cost = -7.067988454154528
Step 25, Cost = -7.081842534715245
Step 26, Cost = -7.095617260802699
Step 27, Cost = -7.109472588274404
Step 28, Cost = -7.123480825409055
Step 29, Cost = -7.137684426026871
Step 30, Cost = -7.152041022693112
Step 31, Cost = -7.166453310287315
Step 32, Cost = -7.1807483416093865
Step 33, Cost = -7.194694917926691
Step 34, Cost = -7.208028603663316
Step 35, Cost = -7.22045639587038
Step 36, Cost = -7.231727330032204
Step 37, Cost = -7.241565955502979
Step 38, Cost = -7.249767410209168
Step 39, Cost = -7.255782895664824
 </code>
 </pre>
 </details>

---

## 25. tutorial_qaoa_intro.html <a name="demo24"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html):

```
[[0.45959941488399797, 0.9609527141073113], [0.2702996191454587, 0.7804239603322595]]
Optimal Parameters
[[0.5980635175924566, 0.9419848542526791], [0.5279728111755442, 0.855528453707565]]
Optimal Parameters
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qaoa_intro.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Optimal Parameters
[[0.5980635175924566, 0.9419848542526791], [0.5279728111755442, 0.8555284537075651]]
```

---

## 26. tutorial_vqe_qng.html <a name="demo25"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_qng.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Iteration = 0,  Energy = 0.51052556 Ha,  Convergence parameter = 0.06664604 Ha
Iteration = 20,  Energy = -0.90729965 Ha,  Convergence parameter = 0.05006082 Ha
Iteration = 40,  Energy = -1.35504644 Ha,  Convergence parameter = 0.00713113 Ha
Iteration = 60,  Energy = -1.40833787 Ha,  Convergence parameter = 0.00072399 Ha
Iteration = 80,  Energy = -1.41364035 Ha,  Convergence parameter = 0.00007078 Ha
Iteration = 100,  Energy = -1.41415774 Ha,  Convergence parameter = 0.00000689 Ha
Final value of the energy = -1.41420585 Ha
Number of iterations =  117
Number of qubits =  4
Iteration = 0,  Energy = -0.09424332 Ha
Iteration = 20,  Energy = -0.55156842 Ha
Iteration = 40,  Energy = -1.12731586 Ha
Iteration = 60,  Energy = -1.13583263 Ha
Iteration = 80,  Energy = -1.13602366 Ha
Iteration = 100,  Energy = -1.13611095 Ha
Iteration = 120,  Energy = -1.13615238 Ha
Final convergence parameter = 0.00000097 Ha
Number of iterations =  130
Final value of the ground-state energy = -1.13616398 Ha
Accuracy with respect to the FCI energy: 0.00002547 Ha (0.01598216 kcal/mol)
Final circuit parameters =
 [3.44829694e+00 6.28318531e+00 3.78727399e+00 3.42360201e+00
 5.09234512e-08 4.05827240e+00 2.74944154e+00 6.07360302e+00
 6.24620659e+00 2.40923412e+00 6.28318531e+00 3.32314479e+00]
Iteration = 0,  Energy = -0.32164518 Ha
Iteration = 4,  Energy = -0.46875033 Ha
Iteration = 8,  Energy = -0.85091055 Ha
Iteration = 12,  Energy = -1.13575339 Ha
Iteration = 16,  Energy = -1.13618916 Ha
Final convergence parameter = 0.00000022 Ha
Number of iterations =  17
Final value of the ground-state energy = -1.13618938 Ha
Accuracy with respect to the FCI energy: 0.00000008 Ha (0.00004854 kcal/mol)
Final circuit parameters =
 [3.44829694e+00 6.28318510e+00 3.78727399e+00 3.42360201e+00
 4.03252161e-04 4.05827240e+00 2.74944154e+00 6.07375181e+00
 6.28402001e+00 2.40923412e+00 6.28318525e+00 3.32314479e+00]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_qng.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/optimize/qng.py:162: UserWarning: The keyword argument diag_approx is deprecated. Please use approx='diag' instead.
Iteration = 0,  Energy = 0.51052556 Ha,  Convergence parameter = 0.06664604 Ha
Iteration = 20,  Energy = -0.90729965 Ha,  Convergence parameter = 0.05006082 Ha
Iteration = 40,  Energy = -1.35504644 Ha,  Convergence parameter = 0.00713113 Ha
Iteration = 60,  Energy = -1.40833787 Ha,  Convergence parameter = 0.00072399 Ha
Iteration = 80,  Energy = -1.41364035 Ha,  Convergence parameter = 0.00007078 Ha
Iteration = 100,  Energy = -1.41415774 Ha,  Convergence parameter = 0.00000689 Ha
Final value of the energy = -1.41420585 Ha
Number of iterations =  117
Number of qubits =  4
Iteration = 0,  Energy = -0.09424332 Ha
Iteration = 20,  Energy = -0.55156842 Ha
Iteration = 40,  Energy = -1.12731586 Ha
Iteration = 60,  Energy = -1.13583263 Ha
Iteration = 80,  Energy = -1.13602366 Ha
Iteration = 100,  Energy = -1.13611095 Ha
Iteration = 120,  Energy = -1.13615238 Ha
Final convergence parameter = 0.00000097 Ha
Number of iterations =  130
Final value of the ground-state energy = -1.13616398 Ha
Accuracy with respect to the FCI energy: 0.00002547 Ha (0.01598216 kcal/mol)
Final circuit parameters =
 [3.44829694e+00 6.28318531e+00 3.78727399e+00 3.42360201e+00
 5.09234513e-08 4.05827240e+00 2.74944154e+00 6.07360302e+00
 6.24620659e+00 2.40923412e+00 6.28318531e+00 3.32314479e+00]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/optimize/qng.py:162: UserWarning: The keyword argument diag_approx is deprecated. Please use approx='diag' instead.
Iteration = 0,  Energy = -0.32164518 Ha
Iteration = 4,  Energy = -0.46875033 Ha
Iteration = 8,  Energy = -0.85091055 Ha
Iteration = 12,  Energy = -1.13575339 Ha
Iteration = 16,  Energy = -1.13618916 Ha
Final convergence parameter = 0.00000022 Ha
Number of iterations =  17
Final value of the ground-state energy = -1.13618938 Ha
Accuracy with respect to the FCI energy: 0.00000008 Ha (0.00004854 kcal/mol)
Final circuit parameters =
 [3.44829694e+00 6.28318510e+00 3.78727399e+00 3.42360201e+00
 </code>
 </pre>
 </details>

---

## 27. tutorial_quantum_natural_gradient.html <a name="demo26"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html):

```
[[ 0.125       0.          0.          0.        ]
 [ 0.          0.1875      0.          0.        ]
 [ 0.          0.          0.24973433 -0.01524701]
 [ 0.          0.         -0.01524701  0.20293623]]
[[0.125      0.         0.         0.        ]
 [0.         0.1875     0.         0.        ]
 [0.         0.         0.24973433 0.        ]
 [0.         0.         0.         0.20293623]]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_natural_gradient.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:195: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
[[ 0.125       0.          0.          0.        ]
 [ 0.          0.1875      0.          0.        ]
 [ 0.          0.          0.24973433 -0.01524701]
 [ 0.          0.         -0.01524701  0.20293623]]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/transforms/metric_tensor.py:164: UserWarning: The keyword argument diag_approx is deprecated. Please use approx='diag' instead.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:195: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
[[0.125      0.         0.         0.        ]
```

---

## 28. tutorial_quantum_transfer_learning.html <a name="demo27"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 39%|###9      | 17.6M/44.7M [00:00<00:00, 184MB/s]
 96%|#########5| 42.8M/44.7M [00:00<00:00, 232MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 225MB/s]
Training started:
Phase: train Epoch: 1/1 Iter: 1/62 Batch time: 0.4330
Phase: train Epoch: 1/1 Iter: 2/62 Batch time: 0.4083
Phase: train Epoch: 1/1 Iter: 3/62 Batch time: 0.3871
Phase: train Epoch: 1/1 Iter: 4/62 Batch time: 0.3865
Phase: train Epoch: 1/1 Iter: 5/62 Batch time: 0.3862
Phase: train Epoch: 1/1 Iter: 6/62 Batch time: 0.3783
Phase: train Epoch: 1/1 Iter: 7/62 Batch time: 0.3855
Phase: train Epoch: 1/1 Iter: 8/62 Batch time: 0.4089
Phase: train Epoch: 1/1 Iter: 9/62 Batch time: 0.4095
Phase: train Epoch: 1/1 Iter: 10/62 Batch time: 0.3852
Phase: train Epoch: 1/1 Iter: 11/62 Batch time: 0.3946
Phase: train Epoch: 1/1 Iter: 12/62 Batch time: 0.4017
Phase: train Epoch: 1/1 Iter: 13/62 Batch time: 0.3919
Phase: train Epoch: 1/1 Iter: 14/62 Batch time: 0.4005
Phase: train Epoch: 1/1 Iter: 15/62 Batch time: 0.4046
Phase: train Epoch: 1/1 Iter: 16/62 Batch time: 0.3828
Phase: train Epoch: 1/1 Iter: 17/62 Batch time: 0.3854
Phase: train Epoch: 1/1 Iter: 18/62 Batch time: 0.3853
Phase: train Epoch: 1/1 Iter: 19/62 Batch time: 0.3840
Phase: train Epoch: 1/1 Iter: 20/62 Batch time: 0.4000
Phase: train Epoch: 1/1 Iter: 21/62 Batch time: 0.4116
Phase: train Epoch: 1/1 Iter: 22/62 Batch time: 0.3861
Phase: train Epoch: 1/1 Iter: 23/62 Batch time: 0.3942
Phase: train Epoch: 1/1 Iter: 24/62 Batch time: 0.4046
Phase: train Epoch: 1/1 Iter: 25/62 Batch time: 0.4046
Phase: train Epoch: 1/1 Iter: 26/62 Batch time: 0.3948
Phase: train Epoch: 1/1 Iter: 27/62 Batch time: 0.3835
Phase: train Epoch: 1/1 Iter: 28/62 Batch time: 0.3957
Phase: train Epoch: 1/1 Iter: 29/62 Batch time: 0.3898
Phase: train Epoch: 1/1 Iter: 30/62 Batch time: 0.3914
Phase: train Epoch: 1/1 Iter: 31/62 Batch time: 0.3787
Phase: train Epoch: 1/1 Iter: 32/62 Batch time: 0.3767
Phase: train Epoch: 1/1 Iter: 33/62 Batch time: 0.3935
Phase: train Epoch: 1/1 Iter: 34/62 Batch time: 0.3951
Phase: train Epoch: 1/1 Iter: 35/62 Batch time: 0.4021
Phase: train Epoch: 1/1 Iter: 36/62 Batch time: 0.3879
Phase: train Epoch: 1/1 Iter: 37/62 Batch time: 0.3815
Phase: train Epoch: 1/1 Iter: 38/62 Batch time: 0.4034
Phase: train Epoch: 1/1 Iter: 39/62 Batch time: 0.4072
Phase: train Epoch: 1/1 Iter: 40/62 Batch time: 0.4113
Phase: train Epoch: 1/1 Iter: 41/62 Batch time: 0.3887
Phase: train Epoch: 1/1 Iter: 42/62 Batch time: 0.3790
Phase: train Epoch: 1/1 Iter: 43/62 Batch time: 0.3853
Phase: train Epoch: 1/1 Iter: 44/62 Batch time: 0.3897
Phase: train Epoch: 1/1 Iter: 45/62 Batch time: 0.3962
Phase: train Epoch: 1/1 Iter: 46/62 Batch time: 0.3955
Phase: train Epoch: 1/1 Iter: 47/62 Batch time: 0.3867
Phase: train Epoch: 1/1 Iter: 48/62 Batch time: 0.3860
Phase: train Epoch: 1/1 Iter: 49/62 Batch time: 0.3906
Phase: train Epoch: 1/1 Iter: 50/62 Batch time: 0.3794
Phase: train Epoch: 1/1 Iter: 51/62 Batch time: 0.3880
Phase: train Epoch: 1/1 Iter: 52/62 Batch time: 0.3989
Phase: train Epoch: 1/1 Iter: 53/62 Batch time: 0.4025
Phase: train Epoch: 1/1 Iter: 54/62 Batch time: 0.3880
Phase: train Epoch: 1/1 Iter: 55/62 Batch time: 0.3874
Phase: train Epoch: 1/1 Iter: 56/62 Batch time: 0.3826
Phase: train Epoch: 1/1 Iter: 57/62 Batch time: 0.3771
Phase: train Epoch: 1/1 Iter: 58/62 Batch time: 0.4031
Phase: train Epoch: 1/1 Iter: 59/62 Batch time: 0.4030
Phase: train Epoch: 1/1 Iter: 60/62 Batch time: 0.3956
Phase: train Epoch: 1/1 Iter: 61/62 Batch time: 0.3876
Phase: train Epoch: 1/1 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/1 Iter: 1/39 Batch time: 0.3166
Phase: validation Epoch: 1/1 Iter: 2/39 Batch time: 0.3112
Phase: validation Epoch: 1/1 Iter: 3/39 Batch time: 0.3307
Phase: validation Epoch: 1/1 Iter: 4/39 Batch time: 0.3179
Phase: validation Epoch: 1/1 Iter: 5/39 Batch time: 0.3191
Phase: validation Epoch: 1/1 Iter: 6/39 Batch time: 0.3211
Phase: validation Epoch: 1/1 Iter: 7/39 Batch time: 0.3276
Phase: validation Epoch: 1/1 Iter: 8/39 Batch time: 0.3216
Phase: validation Epoch: 1/1 Iter: 9/39 Batch time: 0.3127
Phase: validation Epoch: 1/1 Iter: 10/39 Batch time: 0.3112
Phase: validation Epoch: 1/1 Iter: 11/39 Batch time: 0.3162
Phase: validation Epoch: 1/1 Iter: 12/39 Batch time: 0.3222
Phase: validation Epoch: 1/1 Iter: 13/39 Batch time: 0.3221
Phase: validation Epoch: 1/1 Iter: 14/39 Batch time: 0.3353
Phase: validation Epoch: 1/1 Iter: 15/39 Batch time: 0.3280
Phase: validation Epoch: 1/1 Iter: 16/39 Batch time: 0.3271
Phase: validation Epoch: 1/1 Iter: 17/39 Batch time: 0.3157
Phase: validation Epoch: 1/1 Iter: 18/39 Batch time: 0.3151
Phase: validation Epoch: 1/1 Iter: 19/39 Batch time: 0.3124
Phase: validation Epoch: 1/1 Iter: 20/39 Batch time: 0.3152
Phase: validation Epoch: 1/1 Iter: 21/39 Batch time: 0.3277
Phase: validation Epoch: 1/1 Iter: 22/39 Batch time: 0.3251
Phase: validation Epoch: 1/1 Iter: 23/39 Batch time: 0.3270
Phase: validation Epoch: 1/1 Iter: 24/39 Batch time: 0.3148
Phase: validation Epoch: 1/1 Iter: 25/39 Batch time: 0.3395
Phase: validation Epoch: 1/1 Iter: 26/39 Batch time: 0.3320
Phase: validation Epoch: 1/1 Iter: 27/39 Batch time: 0.3386
Phase: validation Epoch: 1/1 Iter: 28/39 Batch time: 0.3382
Phase: validation Epoch: 1/1 Iter: 29/39 Batch time: 0.3358
Phase: validation Epoch: 1/1 Iter: 30/39 Batch time: 0.3251
Phase: validation Epoch: 1/1 Iter: 31/39 Batch time: 0.3419
Phase: validation Epoch: 1/1 Iter: 32/39 Batch time: 0.3464
Phase: validation Epoch: 1/1 Iter: 33/39 Batch time: 0.3431
Phase: validation Epoch: 1/1 Iter: 34/39 Batch time: 0.3343
Phase: validation Epoch: 1/1 Iter: 35/39 Batch time: 0.3312
Phase: validation Epoch: 1/1 Iter: 36/39 Batch time: 0.3311
Phase: validation Epoch: 1/1 Iter: 37/39 Batch time: 0.3428
Phase: validation Epoch: 1/1 Iter: 38/39 Batch time: 0.3566
Phase: validation Epoch: 1/1 Iter: 39/39 Batch time: 0.0951
Phase: validation   Epoch: 1/1 Loss: 0.6432 Acc: 0.6536
Training completed in 0m 39s
Best test loss: 0.6432 | Best test accuracy: 0.6536
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  0%|          | 208k/44.7M [00:00<00:22, 2.08MB/s]
  3%|3         | 1.40M/44.7M [00:00<00:05, 8.17MB/s]
 17%|#7        | 7.79M/44.7M [00:00<00:01, 34.9MB/s]
 36%|###5      | 15.9M/44.7M [00:00<00:00, 54.6MB/s]
 54%|#####3    | 23.9M/44.7M [00:00<00:00, 65.2MB/s]
 72%|#######1  | 32.1M/44.7M [00:00<00:00, 72.1MB/s]
 90%|######### | 40.4M/44.7M [00:00<00:00, 76.4MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 62.1MB/s]
Training started:
Phase: train Epoch: 1/1 Iter: 1/62 Batch time: 0.2012
Phase: train Epoch: 1/1 Iter: 2/62 Batch time: 0.1929
Phase: train Epoch: 1/1 Iter: 3/62 Batch time: 0.1728
Phase: train Epoch: 1/1 Iter: 4/62 Batch time: 0.1726
Phase: train Epoch: 1/1 Iter: 5/62 Batch time: 0.1733
Phase: train Epoch: 1/1 Iter: 6/62 Batch time: 0.1726
Phase: train Epoch: 1/1 Iter: 7/62 Batch time: 0.1734
Phase: train Epoch: 1/1 Iter: 8/62 Batch time: 0.1729
Phase: train Epoch: 1/1 Iter: 9/62 Batch time: 0.1735
Phase: train Epoch: 1/1 Iter: 10/62 Batch time: 0.1737
Phase: train Epoch: 1/1 Iter: 11/62 Batch time: 0.1738
Phase: train Epoch: 1/1 Iter: 12/62 Batch time: 0.1730
Phase: train Epoch: 1/1 Iter: 13/62 Batch time: 0.1735
Phase: train Epoch: 1/1 Iter: 14/62 Batch time: 0.1782
Phase: train Epoch: 1/1 Iter: 15/62 Batch time: 0.1753
Phase: train Epoch: 1/1 Iter: 16/62 Batch time: 0.1761
Phase: train Epoch: 1/1 Iter: 17/62 Batch time: 0.1772
Phase: train Epoch: 1/1 Iter: 18/62 Batch time: 0.1777
Phase: train Epoch: 1/1 Iter: 19/62 Batch time: 0.1835
Phase: train Epoch: 1/1 Iter: 20/62 Batch time: 0.1959
Phase: train Epoch: 1/1 Iter: 21/62 Batch time: 0.1749
Phase: train Epoch: 1/1 Iter: 22/62 Batch time: 0.1758
Phase: train Epoch: 1/1 Iter: 23/62 Batch time: 0.1910
Phase: train Epoch: 1/1 Iter: 24/62 Batch time: 0.1754
Phase: train Epoch: 1/1 Iter: 25/62 Batch time: 0.1775
Phase: train Epoch: 1/1 Iter: 26/62 Batch time: 0.1774
Phase: train Epoch: 1/1 Iter: 27/62 Batch time: 0.1763
Phase: train Epoch: 1/1 Iter: 28/62 Batch time: 0.1749
Phase: train Epoch: 1/1 Iter: 29/62 Batch time: 0.1865
Phase: train Epoch: 1/1 Iter: 30/62 Batch time: 0.1924
Phase: train Epoch: 1/1 Iter: 31/62 Batch time: 0.1810
Phase: train Epoch: 1/1 Iter: 32/62 Batch time: 0.1865
Phase: train Epoch: 1/1 Iter: 33/62 Batch time: 0.1764
Phase: train Epoch: 1/1 Iter: 34/62 Batch time: 0.1760
Phase: train Epoch: 1/1 Iter: 35/62 Batch time: 0.1750
Phase: train Epoch: 1/1 Iter: 36/62 Batch time: 0.1728
Phase: train Epoch: 1/1 Iter: 37/62 Batch time: 0.1723
Phase: train Epoch: 1/1 Iter: 38/62 Batch time: 0.1749
Phase: train Epoch: 1/1 Iter: 39/62 Batch time: 0.1730
Phase: train Epoch: 1/1 Iter: 40/62 Batch time: 0.1758
Phase: train Epoch: 1/1 Iter: 41/62 Batch time: 0.1757
Phase: train Epoch: 1/1 Iter: 42/62 Batch time: 0.1772
Phase: train Epoch: 1/1 Iter: 43/62 Batch time: 0.1783
Phase: train Epoch: 1/1 Iter: 44/62 Batch time: 0.1799
Phase: train Epoch: 1/1 Iter: 45/62 Batch time: 0.1772
Phase: train Epoch: 1/1 Iter: 46/62 Batch time: 0.1930
Phase: train Epoch: 1/1 Iter: 47/62 Batch time: 0.1853
Phase: train Epoch: 1/1 Iter: 48/62 Batch time: 0.1846
Phase: train Epoch: 1/1 Iter: 49/62 Batch time: 0.1851
Phase: train Epoch: 1/1 Iter: 50/62 Batch time: 0.1866
Phase: train Epoch: 1/1 Iter: 51/62 Batch time: 0.1851
Phase: train Epoch: 1/1 Iter: 52/62 Batch time: 0.1834
Phase: train Epoch: 1/1 Iter: 53/62 Batch time: 0.1830
Phase: train Epoch: 1/1 Iter: 54/62 Batch time: 0.1839
Phase: train Epoch: 1/1 Iter: 55/62 Batch time: 0.1852
Phase: train Epoch: 1/1 Iter: 56/62 Batch time: 0.1841
Phase: train Epoch: 1/1 Iter: 57/62 Batch time: 0.2217
Phase: train Epoch: 1/1 Iter: 58/62 Batch time: 0.1839
Phase: train Epoch: 1/1 Iter: 59/62 Batch time: 0.1856
Phase: train Epoch: 1/1 Iter: 60/62 Batch time: 0.1827
Phase: train Epoch: 1/1 Iter: 61/62 Batch time: 0.1824
Phase: train Epoch: 1/1 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/1 Iter: 1/39 Batch time: 0.1388
Phase: validation Epoch: 1/1 Iter: 2/39 Batch time: 0.1361
Phase: validation Epoch: 1/1 Iter: 3/39 Batch time: 0.1357
Phase: validation Epoch: 1/1 Iter: 4/39 Batch time: 0.1349
Phase: validation Epoch: 1/1 Iter: 5/39 Batch time: 0.1325
Phase: validation Epoch: 1/1 Iter: 6/39 Batch time: 0.1330
Phase: validation Epoch: 1/1 Iter: 7/39 Batch time: 0.1320
Phase: validation Epoch: 1/1 Iter: 8/39 Batch time: 0.1328
Phase: validation Epoch: 1/1 Iter: 9/39 Batch time: 0.1338
Phase: validation Epoch: 1/1 Iter: 10/39 Batch time: 0.1316
Phase: validation Epoch: 1/1 Iter: 11/39 Batch time: 0.1326
Phase: validation Epoch: 1/1 Iter: 12/39 Batch time: 0.1336
Phase: validation Epoch: 1/1 Iter: 13/39 Batch time: 0.1326
Phase: validation Epoch: 1/1 Iter: 14/39 Batch time: 0.1321
Phase: validation Epoch: 1/1 Iter: 15/39 Batch time: 0.1328
Phase: validation Epoch: 1/1 Iter: 16/39 Batch time: 0.1321
Phase: validation Epoch: 1/1 Iter: 17/39 Batch time: 0.1375
Phase: validation Epoch: 1/1 Iter: 18/39 Batch time: 0.1336
Phase: validation Epoch: 1/1 Iter: 19/39 Batch time: 0.1325
Phase: validation Epoch: 1/1 Iter: 20/39 Batch time: 0.1325
Phase: validation Epoch: 1/1 Iter: 21/39 Batch time: 0.1344
Phase: validation Epoch: 1/1 Iter: 22/39 Batch time: 0.1334
Phase: validation Epoch: 1/1 Iter: 23/39 Batch time: 0.1337
Phase: validation Epoch: 1/1 Iter: 24/39 Batch time: 0.1340
Phase: validation Epoch: 1/1 Iter: 25/39 Batch time: 0.1325
Phase: validation Epoch: 1/1 Iter: 26/39 Batch time: 0.1322
Phase: validation Epoch: 1/1 Iter: 27/39 Batch time: 0.1327
Phase: validation Epoch: 1/1 Iter: 28/39 Batch time: 0.1339
Phase: validation Epoch: 1/1 Iter: 29/39 Batch time: 0.1331
Phase: validation Epoch: 1/1 Iter: 30/39 Batch time: 0.1321
Phase: validation Epoch: 1/1 Iter: 31/39 Batch time: 0.1326
Phase: validation Epoch: 1/1 Iter: 32/39 Batch time: 0.1335
Phase: validation Epoch: 1/1 Iter: 33/39 Batch time: 0.1337
Phase: validation Epoch: 1/1 Iter: 34/39 Batch time: 0.1328
Phase: validation Epoch: 1/1 Iter: 35/39 Batch time: 0.1337
Phase: validation Epoch: 1/1 Iter: 36/39 Batch time: 0.1491
Phase: validation Epoch: 1/1 Iter: 37/39 Batch time: 0.1419
 </code>
 </pre>
 </details>

---

## 29. tutorial_chemical_reactions.html <a name="demo28"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_chemical_reactions.html):

```
The equilibrium bond length is 1.5 Bohrs
The bond dissociation energy is 0.198772 Hartrees
The activation energy is 0.027504 Hartrees
Ratio of reaction rates is 1948918
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_chemical_reactions.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
The equilibrium bond length is 1.5 Bohrs
The bond dissociation energy is 0.198772 Hartrees
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
```

---

## 30. tutorial_mol_geo_opt.html <a name="demo29"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_mol_geo_opt.html):

```
Step = 0,  E = -1.26094338 Ha,  bond length = 0.96762 A
Step = 4,  E = -1.27360653 Ha,  bond length = 0.97619 A
Step = 8,  E = -1.27437809 Ha,  bond length = 0.98223 A
Step = 12,  E = -1.27443305 Ha,  bond length = 0.98457 A
Step = 16,  E = -1.27443729 Ha,  bond length = 0.98533 A
Step = 20,  E = -1.27443763 Ha,  bond length = 0.98556 A
Step = 24,  E = -1.27443766 Ha,  bond length = 0.98563 A
Final value of the ground-state energy = -1.27443766 Ha
Ground-state equilibrium geometry
symbol    x        y        z
  H    0.0102   0.0442   0.0000
  H    0.9867   1.6303   0.0000
  H    1.8720   -0.0085   0.0000
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_mol_geo_opt.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Step = 0,  E = -1.26094338 Ha,  bond length = 0.96762 A
Step = 4,  E = -1.27360653 Ha,  bond length = 0.97619 A
Step = 8,  E = -1.27437809 Ha,  bond length = 0.98223 A
Step = 12,  E = -1.27443305 Ha,  bond length = 0.98457 A
Step = 16,  E = -1.27443729 Ha,  bond length = 0.98533 A
Step = 20,  E = -1.27443763 Ha,  bond length = 0.98556 A
Step = 24,  E = -1.27443766 Ha,  bond length = 0.98563 A
Final value of the ground-state energy = -1.27443766 Ha
Ground-state equilibrium geometry
symbol    x        y        z
  H    0.0102   0.0442   0.0000
  H    0.9867   1.6303   0.0000
```

---

## 31. tutorial_QGAN.html <a name="demo30"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_QGAN.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost = -0.05727687478065491
Step 5: cost = -0.26348111033439636
Step 10: cost = -0.4273917004466057
Step 15: cost = -0.47261590510606766
Step 20: cost = -0.48406896367669106
Step 25: cost = -0.48946382384747267
Step 30: cost = -0.49281889386475086
Step 35: cost = -0.4949494309257716
Step 40: cost = -0.49627021909691393
Step 45: cost = -0.49707187968306243
Prob(real classified as real):  0.9985871425596997
Prob(fake classified as real):  0.5011128038167953
Step 0: cost = -0.5833386033773422
Step 5: cost = -0.8915732949972153
Step 10: cost = -0.9784244522452354
Step 15: cost = -0.9946483590174466
Step 20: cost = -0.9984995491686277
Step 25: cost = -0.9995636216044659
Step 30: cost = -0.9998718172573717
Step 35: cost = -0.9999619696027366
Step 40: cost = -0.9999888275397097
Step 45: cost = -0.999996672290763
Prob(fake classified as real):  0.99999862746688
Discriminator cost:  0.0014114849071802382
Real Bloch vector: [-0.2169418   0.45048445 -0.86602525]
Generator Bloch vector: [-0.2840465   0.41893208 -0.86244407]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_QGAN.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost = -0.05727699398994446
Step 5: cost = -0.26348118484020233
Step 10: cost = -0.4273917078971863
Step 15: cost = -0.47261589020490646
Step 20: cost = -0.48406901210546494
Step 25: cost = -0.4894639030098915
Step 30: cost = -0.49281900376081467
Step 35: cost = -0.4949493855237961
Step 40: cost = -0.49627020210027695
Step 45: cost = -0.49707192927598953
Prob(real classified as real):  0.9985870718955994
Prob(fake classified as real):  0.5011127963662148
Step 0: cost = -0.583338625729084
Step 5: cost = -0.8915732204914093
Step 10: cost = -0.9784243106842041
Step 15: cost = -0.9946482479572296
Step 20: cost = -0.9984994232654572
Step 25: cost = -0.9995635747909546
Step 30: cost = -0.9998717308044434
Step 35: cost = -0.9999619424343109
Step 40: cost = -0.9999886155128479
Step 45: cost = -0.9999965727329254
Prob(fake classified as real):  0.9999985992908478
Discriminator cost:  0.001411527395248413
Real Bloch vector: [-0.21694186  0.45048442 -0.86602521]
Generator Bloch vector: [-0.28404653  0.41893214 -0.86244416]
 </code>
 </pre>
 </details>

---

## 32. tutorial_measurement_optimize.html <a name="demo31"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_measurement_optimize.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   (-46.463906788688924) [I0]
+ (0.7829661725950183) [Z11]
+ (0.7829661725950184) [Z10]
+ (0.8084581961720491) [Z12]
+ (0.8084581961720494) [Z13]
+ (1.2034402289145647) [Z5]
+ (1.203440228914565) [Z4]
+ (1.3096862988615414) [Z7]
+ (1.3096862988615419) [Z6]
+ (12.41263074211177) [Z0]
+ (12.41263074211177) [Z1]
+ (-8.194261372214103e-06) [Y10 Y12]
+ (-8.194261372214103e-06) [X10 X12]
+ (-1.8540608579962234e-06) [Y5 Y7]
+ (-1.8540608579962234e-06) [X5 X7]
+ (-7.764994118276425e-07) [Y3 Y5]
+ (-7.764994118276425e-07) [X3 X5]
+ (-5.929765816500194e-07) [Y4 Y6]
+ (-5.929765816500194e-07) [X4 X6]
+ (1.6021167404902733e-06) [Y2 Y4]
+ (1.6021167404902733e-06) [X2 X4]
+ (7.954413176181225e-06) [Y11 Y13]
+ (7.954413176181225e-06) [X11 X13]
+ (0.003276971931231685) [Y1 Y3]
+ (0.003276971931231685) [X1 X3]
+ (0.1043306478065142) [Y0 Y2]
+ (0.1043306478065142) [X0 X2]
+ (0.11270386920332202) [Z10 Z12]
+ (0.11270386920332202) [Z11 Z13]
+ (0.11383573679388656) [Z4 Z12]
+ (0.11383573679388656) [Z5 Z13]
+ (0.11952438964682646) [Z6 Z10]
+ (0.11952438964682646) [Z7 Z11]
+ (0.12495807739503231) [Z2 Z4]
+ (0.12495807739503231) [Z3 Z5]
+ (0.12799502492468406) [Z2 Z10]
+ (0.12799502492468406) [Z3 Z11]
+ (0.13401715261963684) [Z6 Z12]
+ (0.13401715261963684) [Z7 Z13]
+ (0.13701191674040747) [Z4 Z6]
+ (0.13701191674040747) [Z5 Z7]
+ (0.13734953064261296) [Z6 Z11]
+ (0.13734953064261296) [Z7 Z10]
+ (0.13739104762683224) [Z2 Z6]
+ (0.13739104762683224) [Z3 Z7]
+ (0.1376687264585256) [Z8 Z10]
+ (0.1376687264585256) [Z9 Z11]
+ (0.14011289865354815) [Z2 Z12]
+ (0.14011289865354815) [Z3 Z13]
+ (0.14138905291942788) [Z10 Z13]
+ (0.14138905291942788) [Z11 Z12]
+ (0.1425799771248574) [Z4 Z11]
+ (0.1425799771248574) [Z5 Z10]
+ (0.1472294321876615) [Z8 Z11]
+ (0.1472294321876615) [Z9 Z10]
+ (0.14899430575065542) [Z4 Z7]
+ (0.14899430575065542) [Z5 Z6]
+ (0.1492635514738886) [Z10 Z11]
+ (0.1496070268444531) [Z4 Z8]
+ (0.1496070268444531) [Z5 Z9]
+ (0.14973486803496916) [Z8 Z12]
+ (0.14973486803496916) [Z9 Z13]
+ (0.1513832716142882) [Z6 Z13]
+ (0.1513832716142882) [Z7 Z12]
+ (0.15337968243314137) [Z2 Z11]
+ (0.15337968243314137) [Z3 Z10]
+ (0.1543574865722362) [Z12 Z13]
+ (0.1556901067175246) [Z2 Z13]
+ (0.1556901067175246) [Z3 Z12]
+ (0.155822690515531) [Z8 Z13]
+ (0.155822690515531) [Z9 Z12]
+ (0.15676396176431007) [Z4 Z9]
+ (0.15676396176431007) [Z5 Z8]
+ (0.15755314797985676) [Z4 Z5]
+ (0.1607976453483858) [Z2 Z5]
+ (0.1607976453483858) [Z3 Z4]
+ (0.1675665326546125) [Z6 Z8]
+ (0.1675665326546125) [Z7 Z9]
+ (0.16853486561579945) [Z2 Z7]
+ (0.16853486561579945) [Z3 Z6]
+ (0.18143991440303853) [Z6 Z9]
+ (0.18143991440303853) [Z7 Z8]
+ (0.18189085790751391) [Z2 Z3]
+ (0.18690820476912573) [Z2 Z9]
+ (0.18690820476912573) [Z3 Z8]
+ (0.19299723935364196) [Z0 Z10]
+ (0.19299723935364196) [Z1 Z11]
+ (0.1939253461327015) [Z6 Z7]
+ (0.19661770890342153) [Z0 Z4]
+ (0.19661770890342153) [Z1 Z5]
+ (0.19936354537360834) [Z0 Z5]
+ (0.19936354537360834) [Z1 Z4]
+ (0.2007286646044172) [Z0 Z11]
+ (0.2007286646044172) [Z1 Z10]
+ (0.21102659849791483) [Z0 Z12]
+ (0.21102659849791483) [Z1 Z13]
+ (0.21631037498631778) [Z0 Z13]
+ (0.21631037498631778) [Z1 Z12]
+ (0.23671080783830437) [Z0 Z2]
+ (0.23671080783830437) [Z1 Z3]
+ (0.24164663936017153) [Z0 Z6]
+ (0.24164663936017153) [Z1 Z7]
+ (0.24853483371314206) [Z0 Z7]
+ (0.24853483371314206) [Z1 Z6]
+ (0.2512944567459171) [Z0 Z3]
+ (0.2512944567459171) [Z1 Z2]
+ (0.27232518306605663) [Z0 Z8]
+ (0.27232518306605663) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.186176373486048) [Z0 Z1]
+ (-1.2260484989220518e-05) [Y4 Z5 Y6]
+ (-1.2260484989220518e-05) [X4 Z5 X6]
+ (-1.2260484989220507e-05) [Y5 Z6 Y7]
+ (-1.2260484989220507e-05) [X5 Z6 X7]
+ (-1.0722312157901746e-05) [Y10 Z11 Y12]
+ (-1.0722312157901746e-05) [X10 Z11 X12]
+ (-1.0722312157901743e-05) [Y11 Z12 Y13]
+ (-1.0722312157901743e-05) [X11 Z12 X13]
+ (-3.887051672988983e-06) [Y3 Z4 Y5]
+ (-3.887051672988983e-06) [X3 Z4 X5]
+ (-3.887051672988981e-06) [Y2 Z3 Y4]
+ (-3.887051672988981e-06) [X2 Z3 X4]
+ (0.1250703257977215) [Y1 Z2 Y3]
+ (0.1250703257977215) [X1 Z2 X3]
+ (0.12507032579772154) [Y0 Z1 Y2]
+ (0.12507032579772154) [X0 Z1 X2]
+ (-0.038314670294803906) [Y4 Y5 X12 X13]
+ (-0.038314670294803906) [X4 X5 Y12 Y13]
+ (-0.036194123559042696) [Y2 Y3 X8 X9]
+ (-0.036194123559042696) [X2 X3 Y8 Y9]
+ (-0.0311438179889672) [Y2 Y3 X6 X7]
+ (-0.0311438179889672) [X2 X3 Y6 Y7]
+ (-0.028685183716105876) [Y10 Y11 X12 X13]
+ (-0.028685183716105876) [X10 X11 Y12 Y13]
+ (-0.025996177598021072) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021072) [X3 Z4 Z5 X7]
+ (-0.02538465750845731) [Y2 Y3 X10 X11]
+ (-0.02538465750845731) [X2 X3 Y10 Y11]
+ (-0.019028242443847217) [Y3 Y4 X11 X12]
+ (-0.019028242443847217) [X3 X4 Y11 Y12]
+ (-0.017825140995786526) [Y6 Y7 X10 X11]
+ (-0.017825140995786526) [X6 X7 Y10 Y11]
+ (-0.017680067952481473) [Y4 Y5 X10 X11]
+ (-0.017680067952481473) [X4 X5 Y10 Y11]
+ (-0.017366118994651358) [Y6 Y7 X12 X13]
+ (-0.017366118994651358) [X6 X7 Y12 Y13]
+ (-0.015577208063976458) [Y2 Y3 X12 X13]
+ (-0.015577208063976458) [X2 X3 Y12 Y13]
+ (-0.014583648907612726) [Y0 Y1 X2 X3]
+ (-0.014583648907612726) [X0 X1 Y2 Y3]
+ (-0.013873381748426047) [Y6 Y7 X8 X9]
+ (-0.013873381748426047) [X6 X7 Y8 Y9]
+ (-0.011982389010247965) [Y4 Y5 X6 X7]
+ (-0.011982389010247965) [X4 X5 Y6 Y7]
+ (-0.011285190200840919) [Y5 X6 X11 Y12]
+ (-0.011285190200840919) [X5 Y6 Y11 X12]
+ (-0.009560705729135905) [Y8 Y9 X10 X11]
+ (-0.009560705729135905) [X8 X9 Y10 Y11]
+ (-0.00812525192138104) [Y1 X2 X8 Y9]
+ (-0.00812525192138104) [Y1 Y2 Y8 Y9]
+ (-0.00812525192138104) [X1 X2 X8 X9]
+ (-0.00812525192138104) [X1 Y2 Y8 X9]
+ (-0.007731425250775241) [Y0 Y1 X10 X11]
+ (-0.007731425250775241) [X0 X1 Y10 Y11]
+ (-0.007156934919856958) [Y4 Y5 X8 X9]
+ (-0.007156934919856958) [X4 X5 Y8 Y9]
+ (-0.00688819435297053) [Y0 Y1 X6 X7]
+ (-0.00688819435297053) [X0 X1 Y6 Y7]
+ (-0.006509361201177232) [Y0 Y1 X8 X9]
+ (-0.006509361201177232) [X0 X1 Y8 Y9]
+ (-0.006087822480561848) [Y8 Y9 X12 X13]
+ (-0.006087822480561848) [X8 X9 Y12 Y13]
+ (-0.005283776488402948) [Y0 Y1 X12 X13]
+ (-0.005283776488402948) [X0 X1 Y12 Y13]
+ (-0.005143391768825117) [Y3 X4 X5 Y6]
+ (-0.005143391768825117) [X3 Y4 Y5 X6]
+ (-0.004684903388155219) [Y1 X2 X6 Y7]
+ (-0.004684903388155219) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155219) [X1 X2 X6 X7]
+ (-0.004684903388155219) [X1 Y2 Y6 X7]
+ (-0.004575007626639206) [Y1 X2 X12 Y13]
+ (-0.004575007626639206) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639206) [X1 X2 X12 X13]
+ (-0.004575007626639206) [X1 Y2 Y12 X13]
+ (-0.004424855449441871) [Y1 X2 X4 Y5]
+ (-0.004424855449441871) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441871) [X1 X2 X4 X5]
+ (-0.004424855449441871) [X1 Y2 Y4 X5]
+ (-0.0034795118903343508) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343508) [X2 Z3 Z5 X6]
+ (-0.0034795118903343508) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343508) [X3 Z4 Z6 X7]
+ (-0.002745836470186818) [Y0 Y1 X4 X5]
+ (-0.002745836470186818) [X0 X1 Y4 Y5]
+ (-0.0017992194936630405) [Y1 X2 X10 Y11]
+ (-0.0017992194936630405) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630405) [X1 X2 X10 X11]
+ (-0.0017992194936630405) [X1 Y2 Y10 X11]
+ (-0.0002921986261110142) [Y7 Y8 X9 X10]
+ (-0.0002921986261110142) [X7 X8 Y9 Y10]
+ (-8.194261372214103e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372214103e-06) [Z10 X11 Z12 X13]
+ (-7.801707500503772e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500503772e-06) [X2 Z3 X4 Z11]
+ (-7.801707500503772e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500503772e-06) [X3 Z4 X5 Z10]
+ (-4.643051068455295e-06) [Y3 X4 X10 Y11]
+ (-4.643051068455295e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068455295e-06) [X3 X4 X10 X11]
+ (-4.643051068455295e-06) [X3 Y4 Y10 X11]
+ (-4.588855155721556e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155721556e-06) [X4 Z5 X6 Z13]
+ (-4.588855155721556e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155721556e-06) [X5 Z6 X7 Z12]
+ (-4.5565692181433625e-06) [Y5 X6 X12 Y13]
+ (-4.5565692181433625e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692181433625e-06) [X5 X6 X12 X13]
+ (-4.5565692181433625e-06) [X5 Y6 Y12 X13]
+ (-3.6945132944056433e-06) [Y4 X5 X11 Y12]
+ (-3.6945132944056433e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132944056433e-06) [X4 X5 X11 X12]
+ (-3.6945132944056433e-06) [X4 Y5 Y11 X12]
+ (-3.3440815565396785e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815565396785e-06) [Z0 X5 Z6 X7]
+ (-3.3440815565396785e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815565396785e-06) [Z1 X4 Z5 X6]
+ (-3.158656432048476e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432048476e-06) [X2 Z3 X4 Z10]
+ (-3.158656432048476e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432048476e-06) [X3 Z4 X5 Z11]
+ (-3.0993492436659705e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492436659705e-06) [Z0 X4 Z5 X6]
+ (-3.0993492436659705e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492436659705e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817061864e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817061864e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817061864e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817061864e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050870883e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050870883e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050870883e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050870883e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832586033e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832586033e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832586033e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832586033e-06) [X5 Z6 X7 Z8]
+ (-1.855120121552647e-06) [Z6 Y10 Z11 Y12]
+ (-1.855120121552647e-06) [Z6 X10 Z11 X12]
+ (-1.855120121552647e-06) [Z7 Y11 Z12 Y13]
+ (-1.855120121552647e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579962234e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579962234e-06) [X4 Z5 X6 Z7]
+ (-1.816303169696923e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169696923e-06) [Z4 X11 Z12 X13]
+ (-1.816303169696923e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169696923e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286012077e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286012077e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286012077e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286012077e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139642616e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139642616e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139642616e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139642616e-06) [Z1 X10 Z11 X12]
+ (-1.5973171978258744e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171978258744e-06) [Z8 X10 Z11 X12]
+ (-1.5973171978258744e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171978258744e-06) [Z9 X11 Z12 X13]
+ (-1.454842449055439e-06) [Y3 X4 X6 Y7]
+ (-1.454842449055439e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842449055439e-06) [X3 X4 X6 X7]
+ (-1.454842449055439e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081771695e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081771695e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081771695e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081771695e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099934016e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099934016e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099934016e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099934016e-06) [X3 Z4 X5 Z6]
+ (-1.1908508083413189e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508083413189e-06) [Z0 X3 Z4 X5]
+ (-1.1908508083413189e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508083413189e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370494322e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370494322e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370494322e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370494322e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423991098e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423991098e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423991098e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423991098e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601535399e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601535399e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601535399e-06) [X6 X7 X11 X12]
+ (-1.0358477601535399e-06) [X6 Y7 Y11 X12]
+ (-9.509249752192169e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249752192169e-07) [Z2 X4 Z5 X6]
+ (-9.509249752192169e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249752192169e-07) [Z3 X5 Z6 X7]
+ (-9.344557776769784e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776769784e-07) [Z8 X11 Z12 X13]
+ (-9.344557776769784e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776769784e-07) [Z9 X10 Z11 X12]
+ (-8.337746754513678e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754513678e-07) [Z0 X2 Z3 X4]
+ (-8.337746754513678e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754513678e-07) [Z1 X3 Z4 X5]
+ (-7.956895372249938e-07) [Y3 X4 X8 Y9]
+ (-7.956895372249938e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372249938e-07) [X3 X4 X8 X9]
+ (-7.956895372249938e-07) [X3 Y4 Y8 X9]
+ (-7.764994118276426e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118276426e-07) [X2 Z3 X4 Z5]
+ (-5.929765816500194e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816500194e-07) [Z4 X5 Z6 X7]
+ (-5.770052995151864e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995151864e-07) [X2 Z3 X4 Z9]
+ (-5.770052995151864e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995151864e-07) [X3 Z4 X5 Z8]
+ (-5.47164774458609e-07) [Y1 Y2 X11 X12]
+ (-5.47164774458609e-07) [X1 X2 Y11 Y12]
+ (-4.838052750814339e-07) [Y5 X6 X8 Y9]
+ (-4.838052750814339e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750814339e-07) [X5 X6 X8 X9]
+ (-4.838052750814339e-07) [X5 Y6 Y8 X9]
+ (-3.5707613288995106e-07) [Y0 X1 X3 Y4]
+ (-3.5707613288995106e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613288995106e-07) [X0 X1 X3 X4]
+ (-3.5707613288995106e-07) [X0 Y1 Y3 X4]
+ (-2.44732312873708e-07) [Y0 X1 X5 Y6]
+ (-2.44732312873708e-07) [Y0 Y1 Y5 Y6]
+ (-2.44732312873708e-07) [X0 X1 X5 X6]
+ (-2.44732312873708e-07) [X0 Y1 Y5 X6]
+ (-2.1990516183021512e-07) [Y2 X3 X5 Y6]
+ (-2.1990516183021512e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516183021512e-07) [X2 X3 X5 X6]
+ (-2.1990516183021512e-07) [X2 Y3 Y5 X6]
+ (-1.933241277048773e-07) [Y1 X2 X3 Y4]
+ (-1.933241277048773e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861598924e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861598924e-07) [X1 Z2 Z3 X5]
+ (1.7379332622899712e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622899712e-07) [X0 Z1 Z3 X4]
+ (1.7379332622899712e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622899712e-07) [X1 Z2 Z4 X5]
+ (1.933241277048773e-07) [Y1 Y2 X3 X4]
+ (1.933241277048773e-07) [X1 X2 Y3 Y4]
+ (2.1868423770980732e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423770980732e-07) [X2 Z3 X4 Z8]
+ (2.1868423770980732e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423770980732e-07) [X3 Z4 X5 Z9]
+ (2.593534390620371e-07) [Y2 Z3 Y4 Z6]
+ (2.593534390620371e-07) [X2 Z3 X4 Z6]
+ (2.593534390620371e-07) [Y3 Z4 Y5 Z7]
+ (2.593534390620371e-07) [X3 Z4 X5 Z7]
+ (3.6060718677510705e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718677510705e-07) [X0 Z1 Z2 X4]
+ (3.6060718677510705e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718677510705e-07) [X1 Z3 Z4 X5]
+ (5.47164774458609e-07) [Y1 X2 X11 Y12]
+ (5.47164774458609e-07) [X1 Y2 Y11 X12]
+ (5.627851911228267e-07) [Y0 X1 X11 Y12]
+ (5.627851911228267e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911228267e-07) [X0 X1 X11 X12]
+ (5.627851911228267e-07) [X0 Y1 Y11 X12]
+ (6.628614201488962e-07) [Y8 X9 X11 Y12]
+ (6.628614201488962e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201488962e-07) [X8 X9 X11 X12]
+ (6.628614201488962e-07) [X8 Y9 Y11 X12]
+ (1.109440759013741e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759013741e-06) [Z2 X11 Z12 X13]
+ (1.109440759013741e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759013741e-06) [Z3 X10 Z11 X12]
+ (1.6021167404902733e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167404902733e-06) [Z2 X3 Z4 X5]
+ (1.87821012470872e-06) [Z4 Y10 Z11 Y12]
+ (1.87821012470872e-06) [Z4 X10 Z11 X12]
+ (1.87821012470872e-06) [Z5 Y11 Z12 Y13]
+ (1.87821012470872e-06) [Z5 X11 Z12 X13]
+ (2.172669101412851e-06) [Y2 X3 X11 Y12]
+ (2.172669101412851e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101412851e-06) [X2 X3 X11 X12]
+ (2.172669101412851e-06) [X2 Y3 Y11 X12]
+ (3.1174479459066215e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479459066215e-06) [X0 Z2 Z3 X4]
+ (3.5390541844812396e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844812396e-06) [X2 Z3 X4 Z12]
+ (3.5390541844812396e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844812396e-06) [X3 Z4 X5 Z13]
+ (4.281913884769367e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884769367e-06) [X4 Z5 X6 Z11]
+ (4.281913884769367e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884769367e-06) [X5 Z6 X7 Z10]
+ (5.275883122028445e-06) [Y3 X4 X12 Y13]
+ (5.275883122028445e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122028445e-06) [X3 X4 X12 X13]
+ (5.275883122028445e-06) [X3 Y4 Y12 X13]
+ (5.9743117133705745e-06) [Y5 X6 X10 Y11]
+ (5.9743117133705745e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117133705745e-06) [X5 X6 X10 X11]
+ (5.9743117133705745e-06) [X5 Y6 Y10 X11]
+ (7.954413176181225e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176181225e-06) [X10 Z11 X12 Z13]
+ (8.814937306509684e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306509684e-06) [X2 Z3 X4 Z13]
+ (8.814937306509684e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306509684e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110142) [Y7 X8 X9 Y10]
+ (0.0002921986261110142) [X7 Y8 Y9 X10]
+ (0.0004956762314915776) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915776) [X2 Z4 Z5 X6]
+ (0.0011059037691896858) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896858) [X0 Z1 X2 Z5]
+ (0.0011059037691896858) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896858) [X1 Z2 X3 Z4]
+ (0.0016638798784907659) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907659) [X2 Z3 Z4 X6]
+ (0.0016638798784907659) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907659) [X3 Z5 Z6 X7]
+ (0.0017560707018412461) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412461) [X0 Z1 X2 Z11]
+ (0.0017560707018412461) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412461) [X1 Z2 X3 Z10]
+ (0.0023262306231580823) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580823) [X0 Z1 X2 Z13]
+ (0.0023262306231580823) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580823) [X1 Z2 X3 Z12]
+ (0.002745836470186818) [Y0 X1 X4 Y5]
+ (0.002745836470186818) [X0 Y1 Y4 X5]
+ (0.0029297686747510607) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510607) [X0 Z1 X2 Z9]
+ (0.0029297686747510607) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510607) [X1 Z2 X3 Z8]
+ (0.0032769719312316847) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316847) [X0 Z1 X2 Z3]
+ (0.0033476175306661766) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661766) [X0 Z1 X2 Z7]
+ (0.0033476175306661766) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661766) [X1 Z2 X3 Z6]
+ (0.0035552901955042864) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042864) [X0 Z1 X2 Z10]
+ (0.0035552901955042864) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042864) [X1 Z2 X3 Z11]
+ (0.005143391768825117) [Y3 Y4 X5 X6]
+ (0.005143391768825117) [X3 X4 Y5 Y6]
+ (0.005283776488402948) [Y0 X1 X12 Y13]
+ (0.005283776488402948) [X0 Y1 Y12 X13]
+ (0.005530759218631556) [Y0 Z1 Y2 Z4]
+ (0.005530759218631556) [X0 Z1 X2 Z4]
+ (0.005530759218631556) [Y1 Z2 Y3 Z5]
+ (0.005530759218631556) [X1 Z2 X3 Z5]
+ (0.006087822480561848) [Y8 X9 X12 Y13]
+ (0.006087822480561848) [X8 Y9 Y12 X13]
+ (0.006509361201177232) [Y0 X1 X8 Y9]
+ (0.006509361201177232) [X0 Y1 Y8 X9]
+ (0.00688819435297053) [Y0 X1 X6 Y7]
+ (0.00688819435297053) [X0 Y1 Y6 X7]
+ (0.007156934919856958) [Y4 X5 X8 Y9]
+ (0.007156934919856958) [X4 Y5 Y8 X9]
+ (0.007731425250775241) [Y0 X1 X10 Y11]
+ (0.007731425250775241) [X0 Y1 Y10 X11]
+ (0.008032520918821395) [Y0 Z1 Y2 Z6]
+ (0.008032520918821395) [X0 Z1 X2 Z6]
+ (0.008032520918821395) [Y1 Z2 Y3 Z7]
+ (0.008032520918821395) [X1 Z2 X3 Z7]
+ (0.009560705729135905) [Y8 X9 X10 Y11]
+ (0.009560705729135905) [X8 Y9 Y10 X11]
+ (0.011055020596132099) [Y0 Z1 Y2 Z8]
+ (0.011055020596132099) [X0 Z1 X2 Z8]
+ (0.011055020596132099) [Y1 Z2 Y3 Z9]
+ (0.011055020596132099) [X1 Z2 X3 Z9]
+ (0.011285190200840919) [Y5 Y6 X11 X12]
+ (0.011285190200840919) [X5 X6 Y11 Y12]
+ (0.01130727400884823) [Y7 Z8 Z9 Y11]
+ (0.01130727400884823) [X7 Z8 Z9 X11]
+ (0.011982389010247965) [Y4 X5 X6 Y7]
+ (0.011982389010247965) [X4 Y5 Y6 X7]
+ (0.013873381748426047) [Y6 X7 X8 Y9]
+ (0.013873381748426047) [X6 Y7 Y8 X9]
+ (0.014583648907612726) [Y0 X1 X2 Y3]
+ (0.014583648907612726) [X0 Y1 Y2 X3]
+ (0.015577208063976458) [Y2 X3 X12 Y13]
+ (0.015577208063976458) [X2 Y3 Y12 X13]
+ (0.017366118994651358) [Y6 X7 X12 Y13]
+ (0.017366118994651358) [X6 Y7 Y12 X13]
+ (0.017680067952481473) [Y4 X5 X10 Y11]
+ (0.017680067952481473) [X4 Y5 Y10 X11]
+ (0.017825140995786526) [Y6 X7 X10 Y11]
+ (0.017825140995786526) [X6 Y7 Y10 X11]
+ (0.019028242443847217) [Y3 X4 X11 Y12]
+ (0.019028242443847217) [X3 Y4 Y11 X12]
+ (0.02538465750845731) [Y2 X3 X10 Y11]
+ (0.02538465750845731) [X2 Y3 Y10 X11]
+ (0.028685183716105876) [Y10 X11 X12 Y13]
+ (0.028685183716105876) [X10 Y11 Y12 X13]
+ (0.029812424517345892) [Y6 Z7 Z8 Y10]
+ (0.029812424517345892) [X6 Z7 Z8 X10]
+ (0.029812424517345892) [Y7 Z9 Z10 Y11]
+ (0.029812424517345892) [X7 Z9 Z10 X11]
+ (0.03010462314345691) [Y6 Z7 Z9 Y10]
+ (0.03010462314345691) [X6 Z7 Z9 X10]
+ (0.03010462314345691) [Y7 Z8 Z10 Y11]
+ (0.03010462314345691) [X7 Z8 Z10 X11]
+ (0.030787505389143967) [Y6 Z8 Z9 Y10]
+ (0.030787505389143967) [X6 Z8 Z9 X10]
+ (0.0311438179889672) [Y2 X3 X6 Y7]
+ (0.0311438179889672) [X2 Y3 Y6 X7]
+ (0.036194123559042696) [Y2 X3 X8 Y9]
+ (0.036194123559042696) [X2 Y3 Y8 X9]
+ (0.038314670294803906) [Y4 X5 X12 Y13]
+ (0.038314670294803906) [X4 Y5 Y12 X13]
+ (0.1043306478065142) [Z0 Y1 Z2 Y3]
+ (0.1043306478065142) [Z0 X1 Z2 X3]
+ (-0.12133276911042255) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042255) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042253) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042253) [X3 Z4 Z5 Z6 X7]
+ (3.2020768791235943e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768791235943e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768791235943e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768791235943e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918927) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918927) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918927) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918927) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329043) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329043) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329043) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329043) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273097) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273097) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273097) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273097) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021072) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021072) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646124) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646124) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646124) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646124) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172993) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172993) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172993) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172993) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613976) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613976) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613976) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613976) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613976) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613976) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613976) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613976) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981921) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981921) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981921) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981921) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688727) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688727) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688727) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688727) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688727) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688727) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688727) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688727) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.00812525192138104) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812525192138104) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00730675992883295) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.00730675992883295) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.00730675992883295) [X4 X5 X7 Z8 Z9 X10]
+ (-0.00730675992883295) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826915) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826915) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826915) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826915) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017331) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017331) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017331) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017331) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825116) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825116) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825116) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825116) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155219) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155219) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.00466862031877629) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.00466862031877629) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639206) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639206) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441871) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441871) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840045) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840045) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840045) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840045) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901087) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901087) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901087) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901087) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025531) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025531) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524667) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524667) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630405) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630405) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369447) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369447) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730573) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730573) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730573) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730573) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125435) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125435) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956421) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956421) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956421) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956421) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880587182e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880587182e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880587182e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880587182e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864499022e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864499022e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864499022e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864499022e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215633599e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215633599e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215633599e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215633599e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.44434467585198e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.44434467585198e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.44434467585198e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.44434467585198e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848513921e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848513921e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848513921e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848513921e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433138686e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433138686e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433138686e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433138686e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117133705745e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117133705745e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122028446e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122028446e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068455295e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068455295e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692181433625e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692181433625e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225583953e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225583953e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594519378273e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594519378273e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132944056425e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132944056425e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130513338e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130513338e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130513338e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130513338e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500130708e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500130708e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483195460794e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483195460794e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483195460794e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483195460794e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283483832127e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283483832127e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283483832127e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283483832127e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463111660597e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463111660597e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112501544e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112501544e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101412851e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101412851e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490554387e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490554387e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886470416e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886470416e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824949132e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824949132e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601535399e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601535399e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372249938e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372249938e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742335119e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742335119e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742335119e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742335119e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201488962e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201488962e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914522852e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914522852e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914522852e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914522852e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574464853e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574464853e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574464853e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574464853e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082769917e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082769917e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082769917e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082769917e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911228267e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911228267e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624518916e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624518916e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624518916e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624518916e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624518916e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624518916e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624518916e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624518916e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750814339e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750814339e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613288995106e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613288995106e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393505254437e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393505254437e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265649832926e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265649832926e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265649832926e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265649832926e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.44732312873708e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.44732312873708e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947819602e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947819602e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947819602e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947819602e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516183021515e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516183021515e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770487726e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770487726e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770487726e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770487726e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209154256577e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209154256577e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209154256577e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209154256577e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176418878e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176418878e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176418878e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176418878e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480436683e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480436683e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480436683e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480436683e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480436683e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480436683e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480436683e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480436683e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480436683e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480436683e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480436683e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480436683e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861598924e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861598924e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599413477e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599413477e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599413477e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599413477e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599413477e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599413477e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599413477e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599413477e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.05744659565202e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.05744659565202e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.05744659565202e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.05744659565202e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133692287e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133692287e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133692287e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133692287e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154256574e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154256574e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154256574e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154256574e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516183021515e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516183021515e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.44732312873708e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.44732312873708e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961188831e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961188831e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961188831e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961188831e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393505254437e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393505254437e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613288995106e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613288995106e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750814339e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750814339e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911228267e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911228267e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201488962e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201488962e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372249938e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372249938e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651749821e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651749821e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651749821e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651749821e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601535399e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601535399e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824949132e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824949132e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216733114e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216733114e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216733114e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216733114e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886470416e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886470416e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490554387e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490554387e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101412851e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101412851e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112501544e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112501544e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479459066215e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479459066215e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463111660597e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463111660597e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500130708e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500130708e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289421769e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289421769e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132944056425e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132944056425e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559400184e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559400184e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692181433625e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692181433625e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068455295e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068455295e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122028446e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122028446e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117133705745e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117133705745e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110142) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110142) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110142) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110142) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915776) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915776) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499368) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499368) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499368) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499368) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125435) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125435) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213704) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213704) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213704) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213704) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440408) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440408) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440408) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440408) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369447) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369447) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630405) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630405) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524667) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524667) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133914) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133914) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133914) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133914) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496507) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496507) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496507) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496507) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441871) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441871) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639206) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639206) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.00466862031877629) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.00466862031877629) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155219) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155219) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342216915) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342216915) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342216915) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342216915) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.0053686593581096075) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.0053686593581096075) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.0053686593581096075) [X2 X3 X7 Z8 Z9 X10]
+ (0.0053686593581096075) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921594) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921594) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921594) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921594) [X5 Z6 X7 X11 Z12 X13]
+ (0.00812525192138104) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812525192138104) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694654) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694654) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694654) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694654) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158483) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158483) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158483) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158483) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671546) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671546) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671546) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671546) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542706) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542706) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542706) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542706) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.01130727400884823) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.01130727400884823) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130919) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130919) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130919) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130919) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226563) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226563) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226563) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226563) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380175) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380175) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380175) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380175) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375654) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375654) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375654) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375654) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173040035) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173040035) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173040035) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173040035) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535568) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535568) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535568) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535568) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535568) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535568) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535568) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535568) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068907) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068907) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068907) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068907) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068907) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068907) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068907) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068907) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149645) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149645) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149645) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149645) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884454) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884454) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884454) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884454) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143967) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143967) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129771) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129771) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780782) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780782) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780782) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780782) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661369) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661369) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661369) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661369) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928374307e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928374307e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277928374306e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928374306e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860069540985e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860069540985e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595086006954098e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086006954098e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274327701378225) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378225) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378225) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378225) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638317) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638317) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638317) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638317) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982182) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982182) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982182) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982182) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322893314) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322893314) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322893314) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322893314) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205294) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205294) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205294) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205294) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719756) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719756) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719756) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719756) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0356083789883125) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0356083789883125) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262472) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262472) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262472) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262472) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190546) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190546) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190546) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190546) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026866) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026866) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026866) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026866) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289087) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289087) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289087) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289087) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469286) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469286) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601316) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601316) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600773) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600773) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600773) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600773) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019028242443847217) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847217) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942885) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942885) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942885) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942885) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179528) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179528) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226563) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226563) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162075) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162075) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172989) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172989) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981921) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981921) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840919) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840919) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962576) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962576) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847342) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847342) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847342) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847342) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023947) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023947) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00730675992883295) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.00730675992883295) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613475) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613475) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017332) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017332) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109608) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109608) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840045) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840045) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832871) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832871) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832871) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832871) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235364) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235364) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235364) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235364) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025531) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806594) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806594) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806594) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806594) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524667) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524667) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524667) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524667) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696497) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696497) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696497) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696497) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696497) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696497) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696497) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696497) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569575576) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569575576) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354938) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354938) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354938) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354938) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880587182e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880587182e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530553357e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530553357e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530553357e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530553357e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879513357e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879513357e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879513357e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879513357e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775138493e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775138493e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775138493e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775138493e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467419003e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467419003e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467419003e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467419003e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669367039e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669367039e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669367039e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669367039e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518338320775e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.4818518338320775e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518338320775e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.4818518338320775e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736476536e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736476536e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736476536e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736476536e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038661957e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038661957e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038661957e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038661957e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.72884314716646e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.72884314716646e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.72884314716646e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.72884314716646e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225583953e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225583953e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594519378273e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594519378273e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429222025e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429222025e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429222025e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429222025e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429222025e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429222025e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429222025e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429222025e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202525423e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202525423e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202525423e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202525423e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156046619748e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156046619748e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156046619748e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156046619748e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220982438354e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220982438354e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220982438354e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220982438354e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365659316e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365659316e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365659316e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365659316e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770472117e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174770472117e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770472117e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174770472117e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676498477e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676498477e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676498477e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676498477e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676498477e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676498477e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676498477e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676498477e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782494913e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782494913e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782494913e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782494913e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288516171e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288516171e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288516171e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288516171e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104000007e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104000007e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104000007e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104000007e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975192291e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975192291e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206990596e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206990596e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774458609e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774458609e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471797187096e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471797187096e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471797187096e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471797187096e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896778480085e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896778480085e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108797462e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108797462e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108797462e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108797462e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393505254437e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505254437e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393505254437e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505254437e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265649832926e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265649832926e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935951872e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935951872e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935951872e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935951872e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947819602e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947819602e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154256574e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154256574e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595652018e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595652018e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095918698e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095918698e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095918698e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095918698e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595652018e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595652018e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350641813928e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350641813928e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350641813928e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350641813928e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553496166e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553496166e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553496166e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553496166e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154256574e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154256574e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947819602e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947819602e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265649832926e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265649832926e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896778480085e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896778480085e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774458609e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774458609e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206990596e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206990596e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975192291e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975192291e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886470416e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886470416e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886470416e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886470416e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.628853243516212e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.628853243516212e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.628853243516212e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.628853243516212e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514632288e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514632288e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514632288e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514632288e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003704076e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003704076e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003704076e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003704076e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003704076e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003704076e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003704076e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003704076e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420191130767e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191130767e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191130767e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191130767e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191130767e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191130767e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420191130767e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191130767e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500130708e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500130708e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500130708e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500130708e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289421769e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289421769e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559400184e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559400184e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880587182e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880587182e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569575576) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569575576) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409323) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409323) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409323) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409323) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005518) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005518) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005518) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005518) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005518) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005518) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005518) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005518) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125435) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125435) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125435) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125435) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907531) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907531) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907531) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496617) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496617) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496617) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496617) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126975) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126975) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126975) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126975) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823477) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823477) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823477) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823477) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823477) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823477) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823477) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823477) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619291) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619291) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619291) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619291) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840045) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840045) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914288) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914288) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914288) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914288) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182533) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182533) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182533) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182533) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660382) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660382) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660382) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660382) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660382) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660382) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660382) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660382) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803845) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803845) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803845) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803845) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076841) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076841) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076841) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076841) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109608) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109608) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839338) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839338) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839338) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839338) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017332) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017332) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960934) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960934) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960934) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960934) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613475) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613475) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.00730675992883295) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00730675992883295) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023947) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023947) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962576) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962576) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840919) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840919) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981921) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981921) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172989) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172989) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162075) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162075) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226563) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226563) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179528) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179528) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847217) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847217) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.04587947078129771) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129771) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156145) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156145) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156145) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156145) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702277) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702277) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702276) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702276) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036484) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036484) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036484) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036484) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863631) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863631) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863631) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863631) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634985) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634985) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634985) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634985) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214003) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214003) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214003) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214003) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0356083789883125) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0356083789883125) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366198) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366198) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366198) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366198) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830113) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830113) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830113) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830113) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469286) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469286) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529065) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529065) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601316) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601316) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131463) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131463) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131463) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131463) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898772) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898772) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898772) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898772) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179528) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179528) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179528) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179528) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.01031148248983187) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983187) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983187) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01031148248983187) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962576) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962576) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962576) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962576) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209827) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209827) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209827) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209827) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454814) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454814) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454814) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454814) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454814) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454814) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454814) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454814) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023947) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023947) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023947) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023947) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.00466862031877629) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00466862031877629) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336936) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336936) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285425) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285425) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285425) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285425) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217892) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217892) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832871) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832871) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235364) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235364) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101627) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101627) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369447) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369447) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312422) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312422) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169508) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169508) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169508) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169508) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024438) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024438) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487723) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487723) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756928) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756928) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354938) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354938) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221159401e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221159401e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221159401e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221159401e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736476536e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736476536e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463111660597e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463111660597e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112501544e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112501544e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063334694e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063334694e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713260843e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713260843e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202525423e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202525423e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562470428e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562470428e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507351828e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507351828e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507351828e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507351828e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102901122e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102901122e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102901122e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102901122e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198957896e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198957896e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198957896e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198957896e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198957896e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198957896e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198957896e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198957896e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985784426e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985784426e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985784426e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985784426e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986241941e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986241941e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986241941e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986241941e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104000008e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104000008e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464688606e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464688606e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464688606e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464688606e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464688606e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464688606e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464688606e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464688606e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422190447e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422190447e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422190447e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422190447e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422190447e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422190447e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422190447e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422190447e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475211098876e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475211098876e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475211098876e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475211098876e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308393932e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308393932e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308393932e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308393932e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308393932e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308393932e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308393932e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308393932e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935951872e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935951872e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815453200623e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815453200623e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553496166e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553496166e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350641813928e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350641813928e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244023268e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244023268e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244023268e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244023268e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244023268e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244023268e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244023268e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244023268e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792272476e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792272476e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253792272476e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253792272476e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655508791e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655508791e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655508791e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655508791e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350641813928e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350641813928e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183387968e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183387968e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183387968e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183387968e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493540178e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493540178e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493540178e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493540178e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553496166e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553496166e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051179164e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051179164e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051179164e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051179164e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815453200623e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815453200623e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935951872e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935951872e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506160373835e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506160373835e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506160373835e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506160373835e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506160373835e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506160373835e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506160373835e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506160373835e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854041097e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854041097e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854041097e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854041097e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150950802427e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150950802427e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150950802427e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150950802427e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425329392e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425329392e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425329392e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425329392e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425329392e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425329392e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425329392e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425329392e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104000008e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104000008e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562470428e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562470428e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202525423e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202525423e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713260843e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713260843e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765759283653e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765759283653e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011590159e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011590159e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011590159e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011590159e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063334694e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063334694e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112501544e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112501544e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463111660597e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463111660597e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671151311e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671151311e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671151311e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671151311e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736476536e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736476536e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721833861e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721833861e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721833861e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721833861e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327398354e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327398354e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327398354e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327398354e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505017648555e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505017648555e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505017648555e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505017648555e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656296861e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656296861e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656296861e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656296861e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717923628e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717923628e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717923628e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717923628e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347890231e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347890231e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793159947e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793159947e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793159947e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793159947e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216056e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411216056e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411216056e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216056e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354938) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354938) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338954708) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338954708) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338954708) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338954708) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756928) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756928) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957558) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957558) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957558) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957558) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487723) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487723) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908498) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908498) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908498) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908498) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024438) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024438) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532483523072988) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532483523072988) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532483523072988) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532483523072988) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312422) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312422) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369447) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369447) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158605) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158605) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158605) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158605) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235364) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235364) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832871) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832871) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217892) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217892) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336936) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336936) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.00466862031877629) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00466862031877629) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00476727218827804) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.00476727218827804) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.00476727218827804) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.00476727218827804) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865465382268126) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865465382268126) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865465382268126) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865465382268126) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224099236) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224099236) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224099236) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224099236) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796775) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796775) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796775) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796775) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908937) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908937) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908937) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908937) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162075) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162075) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162075) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162075) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936375) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936375) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936375) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936375) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936375) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936375) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936375) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936375) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0585919887338615) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0585919887338615) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527021446e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527021446e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527021446e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527021446e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002428) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002428) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002433) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002433) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.01031148248983187) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031148248983187) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209827) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209827) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770621) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770621) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770621) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770621) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676635) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676635) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676635) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676635) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285425) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285425) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121918) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121918) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121918) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121918) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158605) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158605) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093985) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093985) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093985) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093985) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101627) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101627) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587437) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587437) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587437) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587437) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587437) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587437) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587437) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587437) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312422) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312422) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312422) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312422) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538273) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538273) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538273) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538273) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538273) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538273) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538273) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538273) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562583) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562583) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562583) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562583) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146306145267899e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.146306145267899e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713260843e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713260843e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713260843e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713260843e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562470428e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562470428e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562470428e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562470428e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129733362e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129733362e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129733362e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129733362e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229381097e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229381097e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229381097e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229381097e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.10551503654558e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.10551503654558e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.10551503654558e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.10551503654558e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212615551e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212615551e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212615551e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212615551e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413593734e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413593734e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975192291e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975192291e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657776538e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657776538e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657776538e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657776538e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206990596e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206990596e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896778480085e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896778480085e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325313118115e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325313118115e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325313118115e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325313118115e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458843014e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458843014e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599883739886e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599883739886e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599883739886e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599883739886e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754141879e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754141879e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754141879e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754141879e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928355164e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928355164e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931946558e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931946558e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931946558e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931946558e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928355164e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928355164e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815453200623e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815453200623e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815453200623e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815453200623e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458843014e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458843014e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896778480085e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896778480085e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023907895723e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023907895723e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023907895723e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023907895723e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206990596e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206990596e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975192291e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975192291e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413593734e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413593734e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.9494764876617e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.9494764876617e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957657693e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957657693e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957657693e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957657693e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765759283653e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765759283653e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063334694e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063334694e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063334694e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063334694e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347890231e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347890231e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734905417e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734905417e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734905417e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734905417e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692563113e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692563113e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692563113e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692563113e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487723) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487723) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487723) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487723) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024438) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024438) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024438) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024438) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441907) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441907) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441907) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441907) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019244958) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019244958) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019244958) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019244958) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004485) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004485) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004485) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004485) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980184) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980184) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980184) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980184) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980184) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980184) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980184) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980184) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158605) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158605) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285425) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285425) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369364) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369364) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369364) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369364) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046415) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046415) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046415) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046415) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209827) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209827) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031148248983187) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01031148248983187) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0585919887338615) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0585919887338615) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.39870090133412e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.39870090133412e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.39870090133412e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.39870090133412e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217892) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217892) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121918) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121918) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756928) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756928) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452678992e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452678992e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957657693e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957657693e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413593734e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413593734e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413593734e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413593734e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928355164e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928355164e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928355164e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928355164e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588430145e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714588430145e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588430145e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714588430145e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487661701e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487661701e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957657693e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957657693e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756928) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756928) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121918) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121918) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217892) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217892) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XYI =  0.022659767960222288
Expectation value of XIZ =  0.07715357869738898
Expectation value of XYI =  0.022659767960222343
Expectation value of XIZ =  0.07715357869738915
<H> =  3.8768259168631207
3.8768259168631207
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_measurement_optimize.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   (-46.463906788688966) [I0]
+ (0.7829661725950192) [Z10]
+ (0.7829661725950192) [Z11]
+ (0.8084581961720487) [Z12]
+ (0.8084581961720487) [Z13]
+ (1.2034402289145627) [Z5]
+ (1.2034402289145631) [Z4]
+ (1.3096862988615428) [Z7]
+ (1.3096862988615432) [Z6]
+ (12.412630742111766) [Z1]
+ (12.412630742111768) [Z0]
+ (-8.194261371951097e-06) [Y10 Y12]
+ (-8.194261371951097e-06) [X10 X12]
+ (-1.8540608579763888e-06) [Y5 Y7]
+ (-1.8540608579763888e-06) [X5 X7]
+ (-7.764994120219154e-07) [Y3 Y5]
+ (-7.764994120219154e-07) [X3 X5]
+ (-5.929765815814761e-07) [Y4 Y6]
+ (-5.929765815814761e-07) [X4 X6]
+ (1.6021167405102171e-06) [Y2 Y4]
+ (1.6021167405102171e-06) [X2 X4]
+ (7.954413176208913e-06) [Y11 Y13]
+ (7.954413176208913e-06) [X11 X13]
+ (0.0032769719312316474) [Y1 Y3]
+ (0.0032769719312316474) [X1 X3]
+ (0.10433064780651388) [Y0 Y2]
+ (0.10433064780651388) [X0 X2]
+ (0.11270386920332216) [Z10 Z12]
+ (0.11270386920332216) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.11952438964682671) [Z6 Z10]
+ (0.11952438964682671) [Z7 Z11]
+ (0.1249580773950322) [Z2 Z4]
+ (0.1249580773950322) [Z3 Z5]
+ (0.12799502492468418) [Z2 Z10]
+ (0.12799502492468418) [Z3 Z11]
+ (0.13401715261963712) [Z6 Z12]
+ (0.13401715261963712) [Z7 Z13]
+ (0.13701191674040739) [Z4 Z6]
+ (0.13701191674040739) [Z5 Z7]
+ (0.13734953064261318) [Z6 Z11]
+ (0.13734953064261318) [Z7 Z10]
+ (0.13739104762683238) [Z2 Z6]
+ (0.13739104762683238) [Z3 Z7]
+ (0.1376687264585257) [Z8 Z10]
+ (0.1376687264585257) [Z9 Z11]
+ (0.1401128986535483) [Z2 Z12]
+ (0.1401128986535483) [Z3 Z13]
+ (0.1413890529194281) [Z10 Z13]
+ (0.1413890529194281) [Z11 Z12]
+ (0.14257997712485745) [Z4 Z11]
+ (0.14257997712485745) [Z5 Z10]
+ (0.14722943218766163) [Z8 Z11]
+ (0.14722943218766163) [Z9 Z10]
+ (0.14899430575065534) [Z4 Z7]
+ (0.14899430575065534) [Z5 Z6]
+ (0.14926355147388895) [Z10 Z11]
+ (0.14960702684445287) [Z4 Z8]
+ (0.14960702684445287) [Z5 Z9]
+ (0.14973486803496935) [Z8 Z12]
+ (0.14973486803496935) [Z9 Z13]
+ (0.1513832716142885) [Z6 Z13]
+ (0.1513832716142885) [Z7 Z12]
+ (0.1533796824331416) [Z2 Z11]
+ (0.1533796824331416) [Z3 Z10]
+ (0.15435748657223647) [Z12 Z13]
+ (0.1556901067175248) [Z2 Z13]
+ (0.1556901067175248) [Z3 Z12]
+ (0.15582269051553121) [Z8 Z13]
+ (0.15582269051553121) [Z9 Z12]
+ (0.1567639617643098) [Z4 Z9]
+ (0.1567639617643098) [Z5 Z8]
+ (0.15755314797985648) [Z4 Z5]
+ (0.16079764534838567) [Z2 Z5]
+ (0.16079764534838567) [Z3 Z4]
+ (0.1675665326546127) [Z6 Z8]
+ (0.1675665326546127) [Z7 Z9]
+ (0.16853486561579956) [Z2 Z7]
+ (0.16853486561579956) [Z3 Z6]
+ (0.18143991440303878) [Z6 Z9]
+ (0.18143991440303878) [Z7 Z8]
+ (0.18189085790751403) [Z2 Z3]
+ (0.18690820476912567) [Z2 Z9]
+ (0.18690820476912567) [Z3 Z8]
+ (0.19299723935364216) [Z0 Z10]
+ (0.19299723935364216) [Z1 Z11]
+ (0.19392534613270193) [Z6 Z7]
+ (0.19661770890342112) [Z0 Z4]
+ (0.19661770890342112) [Z1 Z5]
+ (0.1993635453736079) [Z0 Z5]
+ (0.1993635453736079) [Z1 Z4]
+ (0.2007286646044174) [Z0 Z11]
+ (0.2007286646044174) [Z1 Z10]
+ (0.2110265984979151) [Z0 Z12]
+ (0.2110265984979151) [Z1 Z13]
+ (0.21631037498631805) [Z0 Z13]
+ (0.21631037498631805) [Z1 Z12]
+ (0.23671080783830428) [Z0 Z2]
+ (0.23671080783830428) [Z1 Z3]
+ (0.24164663936017175) [Z0 Z6]
+ (0.24164663936017175) [Z1 Z7]
+ (0.24853483371314233) [Z0 Z7]
+ (0.24853483371314233) [Z1 Z6]
+ (0.25129445674591694) [Z0 Z3]
+ (0.25129445674591694) [Z1 Z2]
+ (0.2723251830660566) [Z0 Z8]
+ (0.2723251830660566) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.1861763734860475) [Z0 Z1]
+ (-1.2260484988826288e-05) [Y5 Z6 Y7]
+ (-1.2260484988826288e-05) [X5 Z6 X7]
+ (-1.2260484988826287e-05) [Y4 Z5 Y6]
+ (-1.2260484988826287e-05) [X4 Z5 X6]
+ (-1.072231215651305e-05) [Y10 Z11 Y12]
+ (-1.072231215651305e-05) [X10 Z11 X12]
+ (-1.0722312156513047e-05) [Y11 Z12 Y13]
+ (-1.0722312156513047e-05) [X11 Z12 X13]
+ (-3.887051674868822e-06) [Y2 Z3 Y4]
+ (-3.887051674868822e-06) [X2 Z3 X4]
+ (-3.8870516748688214e-06) [Y3 Z4 Y5]
+ (-3.8870516748688214e-06) [X3 Z4 X5]
+ (0.12507032579771954) [Y0 Z1 Y2]
+ (0.12507032579771954) [X0 Z1 X2]
+ (0.12507032579771954) [Y1 Z2 Y3]
+ (0.12507032579771954) [X1 Z2 X3]
+ (-0.03831467029480387) [Y4 Y5 X12 X13]
+ (-0.03831467029480387) [X4 X5 Y12 Y13]
+ (-0.036194123559042675) [Y2 Y3 X8 X9]
+ (-0.036194123559042675) [X2 X3 Y8 Y9]
+ (-0.031143817988967173) [Y2 Y3 X6 X7]
+ (-0.031143817988967173) [X2 X3 Y6 Y7]
+ (-0.028685183716105924) [Y10 Y11 X12 X13]
+ (-0.028685183716105924) [X10 X11 Y12 Y13]
+ (-0.025996177598021194) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021194) [X3 Z4 Z5 X7]
+ (-0.02538465750845742) [Y2 Y3 X10 X11]
+ (-0.02538465750845742) [X2 X3 Y10 Y11]
+ (-0.019028242443847283) [Y3 Y4 X11 X12]
+ (-0.019028242443847283) [X3 X4 Y11 Y12]
+ (-0.017825140995786467) [Y6 Y7 X10 X11]
+ (-0.017825140995786467) [X6 X7 Y10 Y11]
+ (-0.017680067952481542) [Y4 Y5 X10 X11]
+ (-0.017680067952481542) [X4 X5 Y10 Y11]
+ (-0.017366118994651385) [Y6 Y7 X12 X13]
+ (-0.017366118994651385) [X6 X7 Y12 Y13]
+ (-0.015577208063976488) [Y2 Y3 X12 X13]
+ (-0.015577208063976488) [X2 X3 Y12 Y13]
+ (-0.014583648907612655) [Y0 Y1 X2 X3]
+ (-0.014583648907612655) [X0 X1 Y2 Y3]
+ (-0.01387338174842609) [Y6 Y7 X8 X9]
+ (-0.01387338174842609) [X6 X7 Y8 Y9]
+ (-0.011982389010247948) [Y4 Y5 X6 X7]
+ (-0.011982389010247948) [X4 X5 Y6 Y7]
+ (-0.011285190200840914) [Y5 X6 X11 Y12]
+ (-0.011285190200840914) [X5 Y6 Y11 X12]
+ (-0.00956070572913593) [Y8 Y9 X10 X11]
+ (-0.00956070572913593) [X8 X9 Y10 Y11]
+ (-0.008125251921381027) [Y1 X2 X8 Y9]
+ (-0.008125251921381027) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381027) [X1 X2 X8 X9]
+ (-0.008125251921381027) [X1 Y2 Y8 X9]
+ (-0.007731425250775268) [Y0 Y1 X10 X11]
+ (-0.007731425250775268) [X0 X1 Y10 Y11]
+ (-0.007156934919856936) [Y4 Y5 X8 X9]
+ (-0.007156934919856936) [X4 X5 Y8 Y9]
+ (-0.006888194352970556) [Y0 Y1 X6 X7]
+ (-0.006888194352970556) [X0 X1 Y6 Y7]
+ (-0.006509361201177227) [Y0 Y1 X8 X9]
+ (-0.006509361201177227) [X0 X1 Y8 Y9]
+ (-0.006087822480561866) [Y8 Y9 X12 X13]
+ (-0.006087822480561866) [X8 X9 Y12 Y13]
+ (-0.005283776488402959) [Y0 Y1 X12 X13]
+ (-0.005283776488402959) [X0 X1 Y12 Y13]
+ (-0.005143391768825129) [Y3 X4 X5 Y6]
+ (-0.005143391768825129) [X3 Y4 Y5 X6]
+ (-0.00468490338815521) [Y1 X2 X6 Y7]
+ (-0.00468490338815521) [Y1 Y2 Y6 Y7]
+ (-0.00468490338815521) [X1 X2 X6 X7]
+ (-0.00468490338815521) [X1 Y2 Y6 X7]
+ (-0.004575007626639213) [Y1 X2 X12 Y13]
+ (-0.004575007626639213) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639213) [X1 X2 X12 X13]
+ (-0.004575007626639213) [X1 Y2 Y12 X13]
+ (-0.004424855449441847) [Y1 X2 X4 Y5]
+ (-0.004424855449441847) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441847) [X1 X2 X4 X5]
+ (-0.004424855449441847) [X1 Y2 Y4 X5]
+ (-0.0034795118903343265) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343265) [X2 Z3 Z5 X6]
+ (-0.0034795118903343265) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343265) [X3 Z4 Z6 X7]
+ (-0.0027458364701868007) [Y0 Y1 X4 X5]
+ (-0.0027458364701868007) [X0 X1 Y4 Y5]
+ (-0.0017992194936630127) [Y1 X2 X10 Y11]
+ (-0.0017992194936630127) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630127) [X1 X2 X10 X11]
+ (-0.0017992194936630127) [X1 Y2 Y10 X11]
+ (-0.00029219862611104747) [Y7 Y8 X9 X10]
+ (-0.00029219862611104747) [X7 X8 Y9 Y10]
+ (-8.194261371951097e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371951097e-06) [Z10 X11 Z12 X13]
+ (-7.801707500425653e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500425653e-06) [X2 Z3 X4 Z11]
+ (-7.801707500425653e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500425653e-06) [X3 Z4 X5 Z10]
+ (-4.643051068378383e-06) [Y3 X4 X10 Y11]
+ (-4.643051068378383e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068378383e-06) [X3 X4 X10 X11]
+ (-4.643051068378383e-06) [X3 Y4 Y10 X11]
+ (-4.588855155568988e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155568988e-06) [X4 Z5 X6 Z13]
+ (-4.588855155568988e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155568988e-06) [X5 Z6 X7 Z12]
+ (-4.556569218017559e-06) [Y5 X6 X12 Y13]
+ (-4.556569218017559e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218017559e-06) [X5 X6 X12 X13]
+ (-4.556569218017559e-06) [X5 Y6 Y12 X13]
+ (-3.6945132943752094e-06) [Y4 X5 X11 Y12]
+ (-3.6945132943752094e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132943752094e-06) [X4 X5 X11 X12]
+ (-3.6945132943752094e-06) [X4 Y5 Y11 X12]
+ (-3.3440815565030206e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815565030206e-06) [Z0 X5 Z6 X7]
+ (-3.3440815565030206e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815565030206e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320472702e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320472702e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320472702e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320472702e-06) [X3 Z4 X5 Z11]
+ (-3.0993492436104733e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492436104733e-06) [Z0 X4 Z5 X6]
+ (-3.0993492436104733e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492436104733e-06) [Z1 X5 Z6 X7]
+ (-2.890967881501477e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881501477e-06) [Z6 X11 Z12 X13]
+ (-2.890967881501477e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881501477e-06) [Z7 X10 Z11 X12]
+ (-2.177664604685472e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604685472e-06) [Z0 X10 Z11 X12]
+ (-2.177664604685472e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604685472e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832193997e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832193997e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832193997e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832193997e-06) [X5 Z6 X7 Z8]
+ (-1.8551201213412766e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201213412766e-06) [Z6 X10 Z11 X12]
+ (-1.8551201213412766e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201213412766e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579763886e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579763886e-06) [X4 Z5 X6 Z7]
+ (-1.8163031695757546e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031695757546e-06) [Z4 X11 Z12 X13]
+ (-1.8163031695757546e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031695757546e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285126839e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285126839e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285126839e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285126839e-06) [X5 Z6 X7 Z11]
+ (-1.6148794135415951e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794135415951e-06) [Z0 X11 Z12 X13]
+ (-1.6148794135415951e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794135415951e-06) [Z1 X10 Z11 X12]
+ (-1.5973171976135483e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171976135483e-06) [Z8 X10 Z11 X12]
+ (-1.5973171976135483e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171976135483e-06) [Z9 X11 Z12 X13]
+ (-1.4548424491000676e-06) [Y3 X4 X6 Y7]
+ (-1.4548424491000676e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424491000676e-06) [X3 X4 X6 X7]
+ (-1.4548424491000676e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081123334e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081123334e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081123334e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081123334e-06) [X5 Z6 X7 Z9]
+ (-1.1954890101193217e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890101193217e-06) [X2 Z3 X4 Z7]
+ (-1.1954890101193217e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890101193217e-06) [X3 Z4 X5 Z6]
+ (-1.190850808684188e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808684188e-06) [Z0 X3 Z4 X5]
+ (-1.190850808684188e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808684188e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370190079e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370190079e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370190079e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370190079e-06) [Z3 X4 Z5 X6]
+ (-1.0632283421851959e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283421851959e-06) [Z2 X10 Z11 X12]
+ (-1.0632283421851959e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283421851959e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601602003e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601602003e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601602003e-06) [X6 X7 X11 X12]
+ (-1.0358477601602003e-06) [X6 Y7 Y11 X12]
+ (-9.509249751455565e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751455565e-07) [Z2 X4 Z5 X6]
+ (-9.509249751455565e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751455565e-07) [Z3 X5 Z6 X7]
+ (-9.344557774374592e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557774374592e-07) [Z8 X11 Z12 X13]
+ (-9.344557774374592e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557774374592e-07) [Z9 X10 Z11 X12]
+ (-8.33774675737501e-07) [Z0 Y2 Z3 Y4]
+ (-8.33774675737501e-07) [Z0 X2 Z3 X4]
+ (-8.33774675737501e-07) [Z1 Y3 Z4 Y5]
+ (-8.33774675737501e-07) [Z1 X3 Z4 X5]
+ (-7.956895373167892e-07) [Y3 X4 X8 Y9]
+ (-7.956895373167892e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373167892e-07) [X3 X4 X8 X9]
+ (-7.956895373167892e-07) [X3 Y4 Y8 X9]
+ (-7.764994120219155e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120219155e-07) [X2 Z3 X4 Z5]
+ (-5.929765815814761e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815814761e-07) [Z4 X5 Z6 X7]
+ (-5.770052996923285e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996923285e-07) [X2 Z3 X4 Z9]
+ (-5.770052996923285e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996923285e-07) [X3 Z4 X5 Z8]
+ (-5.471647744451605e-07) [Y1 Y2 X11 X12]
+ (-5.471647744451605e-07) [X1 X2 Y11 Y12]
+ (-4.838052751070662e-07) [Y5 X6 X8 Y9]
+ (-4.838052751070662e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751070662e-07) [X5 X6 X8 X9]
+ (-4.838052751070662e-07) [X5 Y6 Y8 X9]
+ (-3.570761329466872e-07) [Y0 X1 X3 Y4]
+ (-3.570761329466872e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329466872e-07) [X0 X1 X3 X4]
+ (-3.570761329466872e-07) [X0 Y1 Y3 X4]
+ (-2.4473231289254754e-07) [Y0 X1 X5 Y6]
+ (-2.4473231289254754e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231289254754e-07) [X0 X1 X5 X6]
+ (-2.4473231289254754e-07) [X0 Y1 Y5 X6]
+ (-2.1990516187345136e-07) [Y2 X3 X5 Y6]
+ (-2.1990516187345136e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516187345136e-07) [X2 X3 X5 X6]
+ (-2.1990516187345136e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771839376e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771839376e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861882278e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861882278e-07) [X1 Z2 Z3 X5]
+ (1.7379332625369994e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332625369994e-07) [X0 Z1 Z3 X4]
+ (1.7379332625369994e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332625369994e-07) [X1 Z2 Z4 X5]
+ (1.9332412771839376e-07) [Y1 Y2 X3 X4]
+ (1.9332412771839376e-07) [X1 X2 Y3 Y4]
+ (2.1868423762446063e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423762446063e-07) [X2 Z3 X4 Z8]
+ (2.1868423762446063e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423762446063e-07) [X3 Z4 X5 Z9]
+ (2.5935343898074565e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343898074565e-07) [X2 Z3 X4 Z6]
+ (2.5935343898074565e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343898074565e-07) [X3 Z4 X5 Z7]
+ (3.6060718681708854e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718681708854e-07) [X0 Z1 Z2 X4]
+ (3.6060718681708854e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718681708854e-07) [X1 Z3 Z4 X5]
+ (5.471647744451605e-07) [Y1 X2 X11 Y12]
+ (5.471647744451605e-07) [X1 Y2 Y11 X12]
+ (5.627851911438769e-07) [Y0 X1 X11 Y12]
+ (5.627851911438769e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911438769e-07) [X0 X1 X11 X12]
+ (5.627851911438769e-07) [X0 Y1 Y11 X12]
+ (6.628614201760889e-07) [Y8 X9 X11 Y12]
+ (6.628614201760889e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201760889e-07) [X8 X9 X11 X12]
+ (6.628614201760889e-07) [X8 Y9 Y11 X12]
+ (1.109440759341723e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759341723e-06) [Z2 X11 Z12 X13]
+ (1.109440759341723e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759341723e-06) [Z3 X10 Z11 X12]
+ (1.6021167405102171e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405102171e-06) [Z2 X3 Z4 X5]
+ (1.8782101247994546e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247994546e-06) [Z4 X10 Z11 X12]
+ (1.8782101247994546e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247994546e-06) [Z5 X11 Z12 X13]
+ (2.172669101526919e-06) [Y2 X3 X11 Y12]
+ (2.172669101526919e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101526919e-06) [X2 X3 X11 X12]
+ (2.172669101526919e-06) [X2 Y3 Y11 X12]
+ (3.1174479463484737e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479463484737e-06) [X0 Z2 Z3 X4]
+ (3.5390541843190684e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541843190684e-06) [X2 Z3 X4 Z12]
+ (3.5390541843190684e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541843190684e-06) [X3 Z4 X5 Z13]
+ (4.281913884806597e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884806597e-06) [X4 Z5 X6 Z11]
+ (4.281913884806597e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884806597e-06) [X5 Z6 X7 Z10]
+ (5.2758831220507256e-06) [Y3 X4 X12 Y13]
+ (5.2758831220507256e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831220507256e-06) [X3 X4 X12 X13]
+ (5.2758831220507256e-06) [X3 Y4 Y12 X13]
+ (5.974311713319281e-06) [Y5 X6 X10 Y11]
+ (5.974311713319281e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713319281e-06) [X5 X6 X10 X11]
+ (5.974311713319281e-06) [X5 Y6 Y10 X11]
+ (7.954413176208911e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176208911e-06) [X10 Z11 X12 Z13]
+ (8.814937306369795e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306369795e-06) [X2 Z3 X4 Z13]
+ (8.814937306369795e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306369795e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611104747) [Y7 X8 X9 Y10]
+ (0.00029219862611104747) [X7 Y8 Y9 X10]
+ (0.000495676231491633) [Y2 Z4 Z5 Y6]
+ (0.000495676231491633) [X2 Z4 Z5 X6]
+ (0.001105903769189672) [Y0 Z1 Y2 Z5]
+ (0.001105903769189672) [X0 Z1 X2 Z5]
+ (0.001105903769189672) [Y1 Z2 Y3 Z4]
+ (0.001105903769189672) [X1 Z2 X3 Z4]
+ (0.0016638798784908032) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908032) [X2 Z3 Z4 X6]
+ (0.0016638798784908032) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908032) [X3 Z5 Z6 X7]
+ (0.0017560707018412327) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412327) [X0 Z1 X2 Z11]
+ (0.0017560707018412327) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412327) [X1 Z2 X3 Z10]
+ (0.002326230623158075) [Y0 Z1 Y2 Z13]
+ (0.002326230623158075) [X0 Z1 X2 Z13]
+ (0.002326230623158075) [Y1 Z2 Y3 Z12]
+ (0.002326230623158075) [X1 Z2 X3 Z12]
+ (0.0027458364701868007) [Y0 X1 X4 Y5]
+ (0.0027458364701868007) [X0 Y1 Y4 X5]
+ (0.0029297686747510408) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510408) [X0 Z1 X2 Z9]
+ (0.0029297686747510408) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510408) [X1 Z2 X3 Z8]
+ (0.0032769719312316474) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316474) [X0 Z1 X2 Z3]
+ (0.003347617530666168) [Y0 Z1 Y2 Z7]
+ (0.003347617530666168) [X0 Z1 X2 Z7]
+ (0.003347617530666168) [Y1 Z2 Y3 Z6]
+ (0.003347617530666168) [X1 Z2 X3 Z6]
+ (0.0035552901955042456) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042456) [X0 Z1 X2 Z10]
+ (0.0035552901955042456) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042456) [X1 Z2 X3 Z11]
+ (0.005143391768825129) [Y3 Y4 X5 X6]
+ (0.005143391768825129) [X3 X4 Y5 Y6]
+ (0.005283776488402959) [Y0 X1 X12 Y13]
+ (0.005283776488402959) [X0 Y1 Y12 X13]
+ (0.005530759218631519) [Y0 Z1 Y2 Z4]
+ (0.005530759218631519) [X0 Z1 X2 Z4]
+ (0.005530759218631519) [Y1 Z2 Y3 Z5]
+ (0.005530759218631519) [X1 Z2 X3 Z5]
+ (0.006087822480561866) [Y8 X9 X12 Y13]
+ (0.006087822480561866) [X8 Y9 Y12 X13]
+ (0.006509361201177227) [Y0 X1 X8 Y9]
+ (0.006509361201177227) [X0 Y1 Y8 X9]
+ (0.006888194352970556) [Y0 X1 X6 Y7]
+ (0.006888194352970556) [X0 Y1 Y6 X7]
+ (0.007156934919856936) [Y4 X5 X8 Y9]
+ (0.007156934919856936) [X4 Y5 Y8 X9]
+ (0.007731425250775268) [Y0 X1 X10 Y11]
+ (0.007731425250775268) [X0 Y1 Y10 X11]
+ (0.008032520918821378) [Y0 Z1 Y2 Z6]
+ (0.008032520918821378) [X0 Z1 X2 Z6]
+ (0.008032520918821378) [Y1 Z2 Y3 Z7]
+ (0.008032520918821378) [X1 Z2 X3 Z7]
+ (0.00956070572913593) [Y8 X9 X10 Y11]
+ (0.00956070572913593) [X8 Y9 Y10 X11]
+ (0.011055020596132066) [Y0 Z1 Y2 Z8]
+ (0.011055020596132066) [X0 Z1 X2 Z8]
+ (0.011055020596132066) [Y1 Z2 Y3 Z9]
+ (0.011055020596132066) [X1 Z2 X3 Z9]
+ (0.011285190200840914) [Y5 Y6 X11 X12]
+ (0.011285190200840914) [X5 X6 Y11 Y12]
+ (0.011307274008848168) [Y7 Z8 Z9 Y11]
+ (0.011307274008848168) [X7 Z8 Z9 X11]
+ (0.011982389010247948) [Y4 X5 X6 Y7]
+ (0.011982389010247948) [X4 Y5 Y6 X7]
+ (0.01387338174842609) [Y6 X7 X8 Y9]
+ (0.01387338174842609) [X6 Y7 Y8 X9]
+ (0.014583648907612655) [Y0 X1 X2 Y3]
+ (0.014583648907612655) [X0 Y1 Y2 X3]
+ (0.015577208063976488) [Y2 X3 X12 Y13]
+ (0.015577208063976488) [X2 Y3 Y12 X13]
+ (0.017366118994651385) [Y6 X7 X12 Y13]
+ (0.017366118994651385) [X6 Y7 Y12 X13]
+ (0.017680067952481542) [Y4 X5 X10 Y11]
+ (0.017680067952481542) [X4 Y5 Y10 X11]
+ (0.017825140995786467) [Y6 X7 X10 Y11]
+ (0.017825140995786467) [X6 Y7 Y10 X11]
+ (0.019028242443847283) [Y3 X4 X11 Y12]
+ (0.019028242443847283) [X3 Y4 Y11 X12]
+ (0.02538465750845742) [Y2 X3 X10 Y11]
+ (0.02538465750845742) [X2 Y3 Y10 X11]
+ (0.028685183716105924) [Y10 X11 X12 Y13]
+ (0.028685183716105924) [X10 Y11 Y12 X13]
+ (0.02981242451734579) [Y6 Z7 Z8 Y10]
+ (0.02981242451734579) [X6 Z7 Z8 X10]
+ (0.02981242451734579) [Y7 Z9 Z10 Y11]
+ (0.02981242451734579) [X7 Z9 Z10 X11]
+ (0.030104623143456834) [Y6 Z7 Z9 Y10]
+ (0.030104623143456834) [X6 Z7 Z9 X10]
+ (0.030104623143456834) [Y7 Z8 Z10 Y11]
+ (0.030104623143456834) [X7 Z8 Z10 X11]
+ (0.030787505389143946) [Y6 Z8 Z9 Y10]
+ (0.030787505389143946) [X6 Z8 Z9 X10]
+ (0.031143817988967173) [Y2 X3 X6 Y7]
+ (0.031143817988967173) [X2 Y3 Y6 X7]
+ (0.036194123559042675) [Y2 X3 X8 Y9]
+ (0.036194123559042675) [X2 Y3 Y8 X9]
+ (0.03831467029480387) [Y4 X5 X12 Y13]
+ (0.03831467029480387) [X4 Y5 Y12 X13]
+ (0.10433064780651388) [Z0 Y1 Z2 Y3]
+ (0.10433064780651388) [Z0 X1 Z2 X3]
+ (-0.12133276911042336) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042336) [X2 Z3 Z4 Z5 X6]
+ (-0.1213327691104233) [Y3 Z4 Z5 Z6 Y7]
+ (-0.1213327691104233) [X3 Z4 Z5 Z6 X7]
+ (3.2020768811409367e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768811409367e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076881140937e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076881140937e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918877) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918877) [X7 Z8 Z9 Z10 X11]
+ (0.2284810656491888) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491888) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823290455) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290455) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290455) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290455) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273117) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273117) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273117) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273117) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021194) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021194) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646166) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646166) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646166) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646166) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172994) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172994) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172994) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172994) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761395) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761395) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761395) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761395) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761395) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761395) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761395) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761395) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819245) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819245) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819245) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819245) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688751) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688751) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688751) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688751) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688751) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688751) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688751) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688751) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381027) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381027) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00730675992883298) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.00730675992883298) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.00730675992883298) [X4 X5 X7 Z8 Z9 X10]
+ (-0.00730675992883298) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826925) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826925) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826925) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826925) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017337) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017337) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017337) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017337) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825129) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825129) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825129) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825129) [X2 Z3 X4 X5 Z6 X7]
+ (-0.00468490338815521) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.00468490338815521) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776299) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776299) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639213) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639213) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441847) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441847) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400835) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400835) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400835) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400835) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901733) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901733) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901733) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901733) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025543) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025543) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524615) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524615) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630127) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630127) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369586) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369586) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730359) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730359) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730359) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730359) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125431) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125431) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956716) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956716) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956716) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956716) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590094e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590094e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590094e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590094e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.77481786441181e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77481786441181e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77481786441181e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77481786441181e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622155268136e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622155268136e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622155268136e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622155268136e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446757460696e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446757460696e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446757460696e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446757460696e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848442144e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848442144e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848442144e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848442144e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.29002843301939e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.29002843301939e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.29002843301939e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.29002843301939e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713319281e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713319281e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122050727e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122050727e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068378383e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068378383e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218017558e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218017558e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.25322422551377e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.25322422551377e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451815635e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451815635e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132943752094e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132943752094e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130432771e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130432771e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130432771e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130432771e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001646395e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001646395e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831953620266e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831953620266e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831953620266e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831953620266e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283482775047e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283482775047e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283482775047e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283482775047e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311057286e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311057286e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711226345e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711226345e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101526919e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101526919e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491000674e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491000674e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886657412e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886657412e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825074232e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825074232e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601602003e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601602003e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373167892e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373167892e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741892245e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741892245e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741892245e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741892245e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201760889e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201760889e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914357503e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914357503e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914357503e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914357503e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574376146e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574376146e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574376146e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574376146e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.92745308237712e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.92745308237712e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.92745308237712e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.92745308237712e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911438769e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911438769e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624485293e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624485293e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624485293e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624485293e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624485293e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624485293e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624485293e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624485293e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751070662e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751070662e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329466872e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329466872e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350707445e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350707445e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565174658e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565174658e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565174658e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565174658e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231289254754e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231289254754e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480603512e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480603512e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480603512e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480603512e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516187345136e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516187345136e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771839376e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771839376e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771839376e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771839376e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155448384e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155448384e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155448384e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155448384e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.551053917596241e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.551053917596241e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.551053917596241e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.551053917596241e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480919645e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480919645e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480919645e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480919645e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480919645e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480919645e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480919645e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480919645e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480919645e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480919645e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480919645e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480919645e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861882278e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861882278e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598983764e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598983764e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598983764e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598983764e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598983764e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598983764e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598983764e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598983764e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595151258e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595151258e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595151258e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595151258e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.64931013692934e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.64931013692934e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.64931013692934e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.64931013692934e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155448384e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155448384e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155448384e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155448384e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516187345136e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516187345136e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231289254754e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231289254754e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961753285e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961753285e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961753285e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961753285e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350707445e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350707445e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329466872e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329466872e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751070662e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751070662e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911438769e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911438769e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201760889e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201760889e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373167892e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373167892e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30653665198699e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30653665198699e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30653665198699e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30653665198699e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601602003e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601602003e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825074232e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825074232e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217161647e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217161647e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217161647e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217161647e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886657412e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886657412e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491000674e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491000674e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101526919e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101526919e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711226345e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711226345e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479463484737e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479463484737e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311057286e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311057286e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001646395e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001646395e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289356139e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289356139e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132943752094e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132943752094e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325593645426e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325593645426e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218017558e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218017558e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068378383e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068378383e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122050727e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122050727e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713319281e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713319281e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611104747) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611104747) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611104747) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611104747) [X6 Z7 X8 X9 Z10 X11]
+ (0.000495676231491633) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.000495676231491633) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499105) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499105) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499105) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499105) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125431) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125431) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213745) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213745) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213745) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213745) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440607) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440607) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440607) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440607) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369586) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369586) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630127) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630127) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524615) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524615) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339174) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339174) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339174) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339174) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496523) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496523) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496523) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496523) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441847) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441847) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639213) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639213) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776299) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776299) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.00468490338815521) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.00468490338815521) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221678) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221678) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221678) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221678) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109568) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109568) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109568) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109568) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921566) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921566) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921566) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921566) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381027) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381027) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694602) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694602) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694602) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694602) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158531) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158531) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158531) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158531) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671559) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671559) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671559) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671559) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542573) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542573) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542573) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542573) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848166) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848166) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130917) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130917) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130917) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130917) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226586) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226586) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226586) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226586) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238021) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238021) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238021) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238021) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375553) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375553) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375553) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375553) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317304001) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317304001) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317304001) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317304001) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535512) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535512) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535512) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535512) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535512) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535512) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535512) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535512) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806896) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806896) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806896) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806896) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806896) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806896) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806896) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806896) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114958) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114958) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114958) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114958) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844555) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844555) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844555) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844555) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143946) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143946) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.0458794707812979) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.0458794707812979) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780763) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780763) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780763) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780763) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661353) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661353) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661353) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661353) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928250463e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928250463e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928250461e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928250461e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086006883061e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006883061e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860068830595e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860068830595e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274327701378293) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378293) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378293) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378293) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.047642612176383076) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383076) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383076) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383076) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982173) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982173) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982173) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982173) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289332) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289332) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289332) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289332) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205311) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205311) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205311) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205311) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197535) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197535) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197535) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197535) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831262) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831262) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262485) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262485) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262485) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262485) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905526) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905526) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905526) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905526) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602681) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602681) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602681) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602681) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890995) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890995) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890995) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890995) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693003) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693003) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952904) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952904) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601291) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601291) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600933) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600933) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600933) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600933) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019028242443847283) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847283) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494289) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494289) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494289) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494289) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179573) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179573) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226586) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226586) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162113) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162113) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172994) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172994) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819245) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840914) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840914) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962635) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962635) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847243) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847243) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847243) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847243) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023918) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023918) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00730675992883298) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.00730675992883298) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561343) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561343) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017337) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017337) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109568) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109568) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400835) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400835) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328805) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328805) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328805) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328805) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423546) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423546) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423546) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423546) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025543) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025543) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066093) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066093) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066093) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066093) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352462) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352462) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352462) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352462) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696503) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696503) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696503) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696503) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696503) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696503) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696503) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696503) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569578937) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569578937) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355042) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730355042) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730355042) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730355042) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880590094e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880590094e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530540666e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530540666e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530540666e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530540666e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879498539e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879498539e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879498539e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879498539e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102774947636e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102774947636e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102774947636e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102774947636e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467425988e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467425988e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467425988e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467425988e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096691677335e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096691677335e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096691677335e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096691677335e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833618225e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833618225e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833618225e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833618225e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736296327e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736296327e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736296327e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736296327e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038651308e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038651308e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038651308e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038651308e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.72884314710022e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.72884314710022e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.72884314710022e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.72884314710022e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225513769e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225513769e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451815635e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451815635e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429224584e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429224584e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429224584e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429224584e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429224584e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429224584e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429224584e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429224584e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203257675e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203257675e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203257675e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203257675e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045864783e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045864783e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045864783e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045864783e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220981010735e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220981010735e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220981010735e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220981010735e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836607283e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836607283e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836607283e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836607283e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770347632e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174770347632e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770347632e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174770347632e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675629362e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675629362e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675629362e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675629362e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675629362e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675629362e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675629362e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675629362e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782507423e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782507423e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782507423e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782507423e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288829217e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288829217e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288829217e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288829217e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104212671e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104212671e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104212671e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104212671e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975120714e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975120714e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.17524620695295e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.17524620695295e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744451605e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744451605e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471799823925e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471799823925e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471799823925e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471799823925e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677602345e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677602345e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231088468246e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231088468246e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231088468246e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231088468246e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350707445e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350707445e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350707445e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350707445e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565174658e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565174658e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595725198e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595725198e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595725198e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595725198e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328948060351e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328948060351e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155448381e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155448381e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595151258e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595151258e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178093982849e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178093982849e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178093982849e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178093982849e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595151258e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595151258e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350648540486e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350648540486e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350648540486e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350648540486e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355495095e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355495095e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355495095e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355495095e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155448381e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155448381e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328948060351e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328948060351e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565174658e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565174658e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677602345e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677602345e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744451605e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744451605e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.17524620695295e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.17524620695295e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975120714e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975120714e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886657412e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886657412e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886657412e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886657412e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434943501e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434943501e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434943501e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434943501e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514146674e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514146674e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514146674e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514146674e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003416627e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003416627e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003416627e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003416627e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003416627e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003416627e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003416627e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003416627e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420189776036e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189776036e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189776036e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189776036e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189776036e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189776036e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420189776036e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189776036e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001646395e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001646395e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001646395e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001646395e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289356139e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289356139e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325593645426e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325593645426e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880590094e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880590094e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569578937) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569578937) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288408456) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288408456) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288408456) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288408456) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005433) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005433) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005433) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005433) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005433) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005433) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005433) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005433) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125431) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125431) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125431) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125431) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907562) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907562) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907562) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907562) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496693) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496693) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496693) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496693) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126945) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126945) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126945) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126945) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482345) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482345) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482345) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482345) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482345) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482345) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482345) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482345) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619304) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619304) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619304) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619304) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400835) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400835) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914303) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914303) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914303) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914303) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182551) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182551) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182551) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182551) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.0051144738316603825) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.0051144738316603825) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.0051144738316603825) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.0051144738316603825) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.0051144738316603825) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603825) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603825) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0051144738316603825) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803861) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803861) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803861) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803861) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076843) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076843) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076843) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076843) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109568) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109568) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839365) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839365) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839365) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839365) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017337) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017337) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960927) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960927) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960927) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960927) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561343) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561343) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.00730675992883298) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00730675992883298) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023918) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023918) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962635) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962635) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840914) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840914) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819245) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172994) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172994) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162113) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162113) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226586) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226586) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179573) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179573) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847283) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847283) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.0458794707812979) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.0458794707812979) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615615) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615615) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615615) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615615) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702292) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702292) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.2816425776702291) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702291) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036466) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036466) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036466) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036466) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863614) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863614) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863614) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863614) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635007) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635007) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635007) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635007) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214023) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214023) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214023) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214023) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831262) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831262) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366184) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366184) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366184) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366184) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829995) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829995) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829995) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829995) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693003) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693003) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529037) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529037) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601291) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601291) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314743) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314743) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314743) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314743) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898907) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898907) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898907) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898907) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917957) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917957) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917957) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917957) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831847) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831847) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831847) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831847) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962635) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962635) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962635) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962635) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209832) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209832) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209832) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209832) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545482) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545482) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545482) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545482) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545482) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545482) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545482) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545482) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023918) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023918) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023918) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023918) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776299) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776299) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336938) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336938) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728533) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728533) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728533) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728533) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178826) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178826) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328805) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328805) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423546) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423546) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015616) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015616) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369586) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369586) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124124) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124124) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168842) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168842) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168842) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168842) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00078708967710244) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.00078708967710244) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487711) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487711) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756787) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756787) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355042) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730355042) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153696e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153696e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153696e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153696e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736296327e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736296327e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311057286e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311057286e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711226345e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711226345e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706416185e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706416185e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713531313e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713531313e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203257675e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203257675e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562525147e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562525147e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650711163e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650711163e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650711163e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650711163e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.35233210280134e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.35233210280134e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.35233210280134e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.35233210280134e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198725242e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198725242e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198725242e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198725242e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198725242e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198725242e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198725242e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198725242e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985672831e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985672831e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985672831e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985672831e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985988732e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985988732e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985988732e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985988732e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104212671e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104212671e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464631507e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464631507e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464631507e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464631507e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464631507e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464631507e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464631507e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464631507e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421995976e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421995976e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421995976e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421995976e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421995976e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421995976e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421995976e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421995976e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521122897e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521122897e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521122897e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521122897e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393083863863e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393083863863e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393083863863e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393083863863e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393083863863e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393083863863e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393083863863e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393083863863e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595725198e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595725198e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815446608224e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815446608224e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355495095e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355495095e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350648540486e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350648540486e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37977324360973e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.37977324360973e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37977324360973e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.37977324360973e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37977324360973e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.37977324360973e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.37977324360973e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.37977324360973e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379217267e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379217267e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379217267e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379217267e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554739104e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554739104e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554739104e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554739104e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350648540486e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350648540486e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183295562e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183295562e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183295562e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183295562e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493866095e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493866095e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493866095e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493866095e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355495095e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355495095e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052031737e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052031737e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052031737e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052031737e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815446608224e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815446608224e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595725198e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595725198e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616003171e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616003171e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616003171e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616003171e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616003171e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616003171e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616003171e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616003171e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.44459785417566e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.44459785417566e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.44459785417566e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.44459785417566e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951892023e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150951892023e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150951892023e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951892023e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425282505e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425282505e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425282505e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425282505e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425282505e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425282505e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425282505e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425282505e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104212671e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104212671e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562525147e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562525147e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203257675e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203257675e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713531313e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713531313e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576024655e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576024655e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011580364e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011580364e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011580364e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011580364e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706416185e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706416185e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711226345e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711226345e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311057286e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311057286e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671192727e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671192727e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671192727e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671192727e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736296327e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736296327e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721924973e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721924973e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721924973e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721924973e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327445241e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327445241e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327445241e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327445241e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501886037e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501886037e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501886037e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501886037e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656352119e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656352119e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656352119e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656352119e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717996549e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717996549e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717996549e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717996549e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733479831465e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733479831465e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793278103e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793278103e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793278103e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793278103e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112169667e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112169667e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112169667e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112169667e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730355042) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730355042) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389552797) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389552797) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389552797) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389552797) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756787) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756787) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569578937) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569578937) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569578937) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569578937) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487711) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487711) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909031) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909031) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909031) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909031) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00078708967710244) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.00078708967710244) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730667) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730667) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730667) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730667) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124124) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124124) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369586) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369586) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415838) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415838) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415838) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415838) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423546) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423546) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328805) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328805) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178826) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178826) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336938) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336938) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776299) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776299) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278122) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278122) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278122) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278122) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226893) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226893) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226893) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226893) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410005) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410005) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410005) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410005) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979677) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979677) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979677) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979677) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01075756395390894) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01075756395390894) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01075756395390894) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01075756395390894) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162113) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162113) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162113) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162113) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936376) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936376) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936376) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936376) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936376) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936376) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936376) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936376) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0585919887338618) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0585919887338618) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527211067e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527211067e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.77595052721107e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.77595052721107e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002677) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002677) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100268) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100268) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.010311482489831846) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831846) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209834) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209834) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297705965) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297705965) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297705965) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297705965) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311869) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311869) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311869) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311869) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311869) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311869) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311869) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311869) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766105) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766105) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766105) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766105) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285325) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285325) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121929) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121929) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121929) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121929) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415838) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415838) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939856) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939856) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939856) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939856) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015616) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015616) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587268) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587268) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587268) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587268) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587268) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587268) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587268) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587268) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124126) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124126) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124126) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124126) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538316) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538316) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538316) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538316) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538316) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538316) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538316) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538316) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856264) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856264) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856264) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856264) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452827267e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452827267e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713531313e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713531313e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713531313e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713531313e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562525147e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562525147e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562525147e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562525147e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297807237e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297807237e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297807237e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297807237e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229770216e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229770216e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229770216e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229770216e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036873794e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036873794e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036873794e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036873794e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212892482e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212892482e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212892482e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212892482e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413603302e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413603302e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975120714e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975120714e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658004816e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658004816e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658004816e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658004816e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.17524620695295e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.17524620695295e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677602345e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677602345e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325320231366e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325320231366e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325320231366e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325320231366e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458880029e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458880029e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884203935e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884203935e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884203935e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884203935e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754808044e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754808044e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754808044e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754808044e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.85056419289642e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.85056419289642e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316290905e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309316290905e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316290905e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309316290905e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.85056419289642e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.85056419289642e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381544660823e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381544660823e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381544660823e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381544660823e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458880029e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458880029e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677602345e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677602345e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023905091187e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023905091187e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023905091187e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023905091187e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.17524620695295e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.17524620695295e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975120714e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975120714e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413603302e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413603302e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487226364e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487226364e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576690525e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576690525e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576690525e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576690525e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576024655e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576024655e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706416185e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706416185e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706416185e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706416185e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733479831465e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2532733479831465e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735044146e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735044146e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735044146e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735044146e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692713197e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692713197e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692713197e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692713197e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487711) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487711) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487711) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487711) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024399) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024399) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024399) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024399) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441846) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441846) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441846) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441846) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245587) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245587) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245587) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245587) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500448) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500448) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500448) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500448) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798016) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798016) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798016) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798016) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798016) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798016) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798016) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798016) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415838) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415838) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285325) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285325) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369386) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369386) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369386) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369386) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046488) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046488) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046488) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046488) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209834) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209834) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831846) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831846) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0585919887338618) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0585919887338618) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014501661e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014501661e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009014501658e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009014501658e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178826) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178826) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121929) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121929) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756787) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756787) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452827267e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452827267e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576690525e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576690525e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413603302e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413603302e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413603302e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413603302e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.85056419289642e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.85056419289642e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.85056419289642e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.85056419289642e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458880029e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458880029e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458880029e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458880029e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487226363e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487226363e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576690525e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576690525e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756787) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756787) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121929) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121929) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178826) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178826) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XYI =  0.022659767960222316
Expectation value of XIZ =  0.07715357869738937
Expectation value of XYI =  0.02265976796022237
Expectation value of XIZ =  0.07715357869738931
<H> =  3.87682591686312
3.87682591686312
 </code>
 </pre>
 </details>

---

## 33. tutorial_expressivity_fourier_series.html <a name="demo32"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.04735694890119547
Cost at step  20: 0.041934267103254284
Cost at step  30: 0.005607479361325123
Cost at step  40: 0.004608455923986271
Cost at step  50: 0.0016064517040624748
Cost at step  10: 0.022632957627087644
Cost at step  20: 0.0018885270619882096
Cost at step  30: 0.0017982806807014667
Cost at step  40: 0.0007504225639153721
Cost at step  50: 0.0006664901287581445
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.04735694890119549
Cost at step  20: 0.04193426710325421
Cost at step  30: 0.0056074793613250925
Cost at step  40: 0.004608455923986286
Cost at step  50: 0.0016064517040624549
Cost at step  10: 0.022632957627087877
Cost at step  20: 0.0018885270619880335
Cost at step  30: 0.0017982806807004612
Cost at step  40: 0.000750422563915062
Cost at step  50: 0.0006664901287574519
```

---

## 34. tutorial_vqt.html <a name="demo33"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqt.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost at Step 0: -0.6605354666522021
Cost at Step 50: -2.869994162243926
Cost at Step 100: -4.613909153498124
Cost at Step 150: -5.395218651684679
Cost at Step 200: -6.189391586583876
Cost at Step 250: -6.552872462856987
Cost at Step 300: -7.096509881592208
Cost at Step 350: -8.216701398205721
Cost at Step 400: -8.085081102568381
Cost at Step 450: -9.388387326738926
Cost at Step 500: -10.019110914765124
Cost at Step 550: -10.630883389198166
Cost at Step 600: -11.181775960997124
Cost at Step 650: -11.48130837267363
Cost at Step 700: -11.723339262895886
Cost at Step 750: -11.82039175189276
Cost at Step 800: -12.314068517894112
Cost at Step 850: -12.594046429266381
Cost at Step 900: -12.824898763081425
Cost at Step 950: -13.032167226825782
Cost at Step 1000: -13.314631473739249
Cost at Step 1050: -13.784075355898747
Cost at Step 1100: -13.975393483108094
Cost at Step 1150: -14.099053330169621
Cost at Step 1200: -14.173584140943465
Cost at Step 1250: -14.244342190620488
Cost at Step 1300: -14.359119617194306
Cost at Step 1350: -14.428516933647717
Cost at Step 1400: -14.513708424970076
Cost at Step 1450: -14.56272566025497
Cost at Step 1500: -14.664132869774456
Cost at Step 1550: -14.706997859130754
Trace Distance: 0.07469762659403273
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqt.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost at Step 0: -0.660535466652201
Cost at Step 50: -2.869994162243927
Cost at Step 100: -4.642067184244081
Cost at Step 150: -5.127022428216539
Cost at Step 200: -6.5292479970263475
Cost at Step 250: -7.026618536189606
Cost at Step 300: -7.488102103496896
Cost at Step 350: -8.746171514554208
Cost at Step 400: -9.427226863807505
Cost at Step 450: -9.53755662322931
Cost at Step 500: -10.388988996560775
Cost at Step 550: -11.109731977694457
Cost at Step 600: -11.250258039584036
Cost at Step 650: -12.17341222302523
Cost at Step 700: -12.491587447630017
Cost at Step 750: -12.804392175167006
Cost at Step 800: -13.051998470031808
Cost at Step 850: -13.121929958625845
Cost at Step 900: -13.257288210600661
Cost at Step 950: -13.37946499688237
Cost at Step 1000: -13.564827616216016
Cost at Step 1050: -13.648718474396574
Cost at Step 1100: -13.83338377080432
Cost at Step 1150: -13.95826726516021
Cost at Step 1200: -14.088858350361166
Cost at Step 1250: -14.062345740176115
Cost at Step 1300: -14.193332801647184
Cost at Step 1350: -14.25625802821952
Cost at Step 1400: -14.26173795086304
Cost at Step 1450: -14.375809556816055
Cost at Step 1500: -14.3895964168573
Cost at Step 1550: -14.503528638687158
Trace Distance: 0.09990470891807307
 </code>
 </pre>
 </details>

---

## 35. tutorial_quantum_chemistry.html <a name="demo34"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
(-46.46390678868894+0j) [] +
(-0.014583648907612688+0j) [X0 X1 Y2 Y3] +
(-3.570761328912313e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017332+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209815+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.792493957714651e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761328912313e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017331+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209815+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.792493957714651e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186808+0j) [X0 X1 Y4 Y5] +
(-2.4473231287016284e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104177005e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285333+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231287016284e-07+0j) [X0 X1 X5 X6] +
(-7.867765104177005e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285333+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970526+0j) [X0 X1 Y6 Y7] +
(-7.735036880587669e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783554428022e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880587669e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783554428022e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177221+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775255+0j) [X0 X1 Y10 Y11] +
(5.627851911474331e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911474331e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402946+0j) [X0 X1 Y12 Y13] +
(0.014583648907612688+0j) [X0 Y1 Y2 X3] +
(3.570761328912313e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017332+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209815+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.792493957714651e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761328912313e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017331+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209815+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.792493957714651e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186808+0j) [X0 Y1 Y4 X5] +
(2.4473231287016284e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104177005e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285333+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231287016284e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104177005e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285333+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970526+0j) [X0 Y1 Y6 X7] +
(7.735036880587669e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783554428022e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880587669e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783554428022e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177221+0j) [X0 Y1 Y8 X9] +
(0.007731425250775255+0j) [X0 Y1 Y10 X11] +
(-5.627851911474331e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911474331e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402946+0j) [X0 Y1 Y12 X13] +
(0.12507032579771885+0j) [X0 Z1 X2] +
(-1.93324127702178e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.002293956611352461+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124237+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714589804297e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.93324127702178e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.002293956611352461+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124237+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714589804297e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231711+0j) [X0 Z1 X2 Z3] +
(-1.5510539175866676e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376507699993e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770608+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480056397e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986461817e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676629+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631569+0j) [X0 Z1 X2 Z4] +
(-1.3807781480056397e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393084446776e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587407+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480056397e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393084446776e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587407+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897164+0j) [X0 Z1 X2 Z5] +
(0.005708495985960949+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332103183108e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.974225379362598e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076848+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.07430598605403e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821402+0j) [X0 Z1 X2 Z6] +
(0.000594022154300555+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773244299948e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.000594022154300555+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244299948e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661952+0j) [X0 Z1 X2 Z7] +
(0.011055020596132118+0j) [X0 Z1 X2 Z8] +
(0.002929768674751091+0j) [X0 Z1 X2 Z9] +
(-6.418291574663824e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914614426e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504313+0j) [X0 Z1 X2 Z10] +
(-1.1076325599378271e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599378271e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018412782+0j) [X0 Z1 X2 Z11] +
(0.006901238249797309+0j) [X0 Z1 X2 Z12] +
(0.0023262306231581027+0j) [X0 Z1 X2 Z13] +
(-3.5682475212381734e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939783+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716555437628e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128841006+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253793111647e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441852+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389678017138e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217887+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199255312e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311865+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881552074+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776294+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975433093e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660395+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464944055e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381025+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630346+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744665312e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624737284e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639206+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441852+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389678017138e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217887+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199255312e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311865+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0046849033881552074+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776294+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975433093e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660395+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464944055e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381025+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630346+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744665312e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624737284e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639206+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768791916496e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125395+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024396+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125395+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024396+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862485524e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.44459785405348e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.001172634831644192+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150951439295e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.002200964069500445+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.839420915415998e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616076527e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798015+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616076527e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798015+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961075709e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310132417822e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126908+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619299+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742655817e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.002261966062482338+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.002261966062482338+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453083016666e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217039554e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536651860109e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.001028329237856253+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806609+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.839420915415998e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.0001940085702975701+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538234+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478339263e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446596391506e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369612+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696477+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565179445e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.839420915415998e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.0001940085702975701+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538234+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289478339263e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446596391506e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369612+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696477+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565179445e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.042743277013781514+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487593+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641928493757e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487593+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641928493757e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255276+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182528+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.001280306097349659+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943052637742e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282183883186e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839362+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425539141e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425539141e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.0052415353828038445+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914283+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907475+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287494106965e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.003356670563832869+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.0001384017730355182+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207150823e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422210716e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423535+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.003356670563832869+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.0001384017730355182+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207150823e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422210716e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423535+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336924+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413756078e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336924+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413756078e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002549+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231015655+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046467+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245578+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219087+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219087+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009015319671e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476487193394e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658554652e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.66134721335648e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230730671+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998845792574e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409991+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298335337e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.0047672721882781165+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037469871e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226877+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079230319246e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001609531381721368+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221152638e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317547659094e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339074+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.000715673424890913+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325318945607e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.60607186769027e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496493+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389555074+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309312072207e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332622604617e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440317+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214168734+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.67040239018765e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651413+0j) [X0 X2] +
(3.117447945899286e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129777+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861664+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453178645e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612688+0j) [Y0 X1 X2 Y3] +
(3.570761328912313e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017332+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209815+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.792493957714651e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761328912313e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017331+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209815+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.792493957714651e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186808+0j) [Y0 X1 X4 Y5] +
(2.4473231287016284e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104177005e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285333+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231287016284e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104177005e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285333+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970526+0j) [Y0 X1 X6 Y7] +
(7.735036880587669e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783554428022e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880587669e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783554428022e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177221+0j) [Y0 X1 X8 Y9] +
(0.007731425250775255+0j) [Y0 X1 X10 Y11] +
(-5.627851911474331e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911474331e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402946+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612688+0j) [Y0 Y1 X2 X3] +
(-3.570761328912313e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017332+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209815+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.792493957714651e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761328912313e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017331+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209815+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.792493957714651e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186808+0j) [Y0 Y1 X4 X5] +
(-2.4473231287016284e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104177005e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285333+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231287016284e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104177005e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285333+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970526+0j) [Y0 Y1 X6 X7] +
(-7.735036880587669e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783554428022e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880587669e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783554428022e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177221+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775255+0j) [Y0 Y1 X10 X11] +
(5.627851911474331e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911474331e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402946+0j) [Y0 Y1 X12 X13] +
(-3.5682475212381734e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939783+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128841006+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253793111647e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716555437628e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579771885+0j) [Y0 Z1 Y2] +
(-1.93324127702178e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.002293956611352461+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124237+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714589804297e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.93324127702178e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.002293956611352461+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124237+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714589804297e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231711+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781480056397e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986461817e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676629+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539175866676e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376507699993e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770608+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631569+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480056397e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393084446776e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587407+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480056397e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393084446776e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587407+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897164+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076848+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.07430598605403e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960949+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.974225379362598e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332103183108e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821402+0j) [Y0 Z1 Y2 Z6] +
(0.000594022154300555+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773244299948e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.000594022154300555+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244299948e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661952+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132118+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674751091+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914614426e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574663824e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504313+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599378271e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599378271e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018412782+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797309+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231581027+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441852+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389678017138e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217887+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199255312e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311865+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0046849033881552074+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776294+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975433093e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660395+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464944055e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381025+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630346+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744665312e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624737284e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639206+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441852+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389678017138e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217887+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199255312e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311865+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881552074+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776294+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975433093e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660395+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464944055e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381025+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630346+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744665312e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624737284e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639206+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.001028329237856253+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806609+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768791916496e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125395+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024396+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125395+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024396+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862485524e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150951439295e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.002200964069500445+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.44459785405348e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.001172634831644192+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.839420915415998e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616076527e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798015+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616076527e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798015+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961075709e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310132417822e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619299+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126908+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742655817e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.002261966062482338+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.002261966062482338+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453083016666e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217039554e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536651860109e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.839420915415998e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.0001940085702975701+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538234+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289478339263e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446596391506e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369612+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696477+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565179445e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.839420915415998e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.0001940085702975701+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538234+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478339263e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446596391506e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369612+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696477+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565179445e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287494106965e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.042743277013781514+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487593+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641928493757e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487593+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641928493757e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255276+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182528+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.001280306097349659+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282183883186e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943052637742e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839362+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425539141e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425539141e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.0052415353828038445+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914283+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907475+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.003356670563832869+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.0001384017730355182+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207150823e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422210716e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423535+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.003356670563832869+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.0001384017730355182+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207150823e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422210716e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423535+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336924+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413756078e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336924+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413756078e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002549+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231015655+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046467+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245578+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219087+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219087+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009015319671e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476487193394e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658554652e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.66134721335648e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230730671+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998845792574e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409991+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298335337e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.0047672721882781165+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037469871e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226877+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079230319246e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001609531381721368+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221152638e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317547659094e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339074+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.000715673424890913+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325318945607e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.60607186769027e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496493+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389555074+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309312072207e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332622604617e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440317+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214168734+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.67040239018765e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651413+0j) [Y0 Y2] +
(3.117447945899286e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129777+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861664+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453178645e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111762+0j) [Z0] +
(0.10433064780651413+0j) [Z0 X1 Z2 X3] +
(3.117447945899286e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129777+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.05859198873386167+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453178645e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651413+0j) [Z0 Y1 Z2 Y3] +
(3.117447945899286e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129777+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.05859198873386167+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453178645e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860482+0j) [Z0 Z1] +
(-8.337746753703554e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273242+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214007+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735385402e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746753703554e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273242+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214007+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735385402e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830448+0j) [Z0 Z2] +
(-1.1908508082615868e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329057+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950634989+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603693100054e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508082615868e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329057+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950634989+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603693100054e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.25129445674591716+0j) [Z0 Z3] +
(-3.099349243447143e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808795441035e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.099349243447143e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808795441035e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.1966177089034213+0j) [Z0 Z4] +
(-3.3440815563173056e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585305858737e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.3440815563173056e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585305858737e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.19936354537360815+0j) [Z0 Z5] +
(0.05608468124661379+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.6522096695973055e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661379+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.6522096695973055e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2416466393601717+0j) [Z0 Z6] +
(0.05600733087780791+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851834053025e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780791+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851834053025e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314222+0j) [Z0 Z7] +
(0.27232518306605663+0j) [Z0 Z8] +
(0.27883454426723386+0j) [Z0 Z9] +
(-2.1776646050417377e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646050417377e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364235+0j) [Z0 Z10] +
(-1.6148794138943045e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794138943045e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441763+0j) [Z0 Z11] +
(0.21102659849791494+0j) [Z0 Z12] +
(0.2163103749863179+0j) [Z0 Z13] +
(1.93324127702178e-07+0j) [X1 X2 Y3 Y4] +
(0.0022939566113524615+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124239+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714589804297e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441852+0j) [X1 X2 X4 X5] +
(-8.091637199255312e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311865+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389678017138e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217887+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881552074+0j) [X1 X2 X6 X7] +
(0.005114473831660395+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464944055e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776294+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975433093e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381025+0j) [X1 X2 X8 X9] +
(-0.0017992194936630344+0j) [X1 X2 X10 X11] +
(-5.287660624737284e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744665312e-07+0j) [X1 X2 Y11 Y12] +
(-0.0045750076266392065+0j) [X1 X2 X12 X13] +
(-1.93324127702178e-07+0j) [X1 Y2 Y3 X4] +
(-0.0022939566113524615+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124239+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714589804297e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441852+0j) [X1 Y2 Y4 X5] +
(-8.091637199255312e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311865+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389678017138e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217887+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881552074+0j) [X1 Y2 Y6 X7] +
(0.005114473831660395+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464944055e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776294+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975433093e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381025+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630344+0j) [X1 Y2 Y10 X11] +
(-5.287660624737284e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744665312e-07+0j) [X1 Y2 Y11 X12] +
(-0.0045750076266392065+0j) [X1 Y2 Y12 X13] +
(0.1250703257977188+0j) [X1 Z2 X3] +
(-1.3807781480056397e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393084446776e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587407+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480056397e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393084446776e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587407+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897164+0j) [X1 Z2 X3 Z4] +
(-1.5510539175866676e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376507699993e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770608+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480056397e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986461817e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676629+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631569+0j) [X1 Z2 X3 Z5] +
(0.000594022154300555+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773244299948e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.000594022154300555+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244299948e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661952+0j) [X1 Z2 X3 Z6] +
(0.005708495985960949+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332103183108e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.974225379362598e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076848+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.07430598605403e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821402+0j) [X1 Z2 X3 Z7] +
(0.002929768674751091+0j) [X1 Z2 X3 Z8] +
(0.011055020596132118+0j) [X1 Z2 X3 Z9] +
(-1.1076325599378271e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599378271e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018412782+0j) [X1 Z2 X3 Z10] +
(-6.418291574663824e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914614426e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504313+0j) [X1 Z2 X3 Z11] +
(0.0023262306231581027+0j) [X1 Z2 X3 Z12] +
(0.006901238249797309+0j) [X1 Z2 X3 Z13] +
(-3.5682475212381734e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939783+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716555437628e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128841006+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253793111647e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125394+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024396+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.839420915415998e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538234+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0001940085702975701+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478339263e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446596391506e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696477+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369612+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565179445e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125394+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024396+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.839420915415998e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538234+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0001940085702975701+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478339263e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446596391506e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696477+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369612+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565179445e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768791916487e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616076527e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798015+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616076527e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798015+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.44459785405348e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.001172634831644192+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150951439295e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.002200964069500445+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.839420915415998e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310132417822e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961075709e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.002261966062482338+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.002261966062482338+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453083016666e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126908+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619299+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742655817e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536651860109e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217039554e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.001028329237856253+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806609+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487593+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641928493757e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832869+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.0001384017730355182+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422210716e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207150823e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423535+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487593+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641928493757e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832869+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.0001384017730355182+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422210716e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207150823e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423535+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.042743277013781514+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.001280306097349659+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182528+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425539141e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425539141e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.0052415353828038445+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943052637742e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282183883186e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839362+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907475+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914283+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287494106965e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336924+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413756078e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336924+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413756078e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219095+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219095+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002553+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245578+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046467+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009015319671e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476487193394e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.66134721335648e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015655+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658554652e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409991+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298335337e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230730671+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998845792574e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226877+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079230319246e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025528+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.0047672721882781165+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037469871e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339074+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.000715673424890913+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325318945607e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.291969486248552e-07+0j) [X1 Z2 Z3 X5] +
(0.001609531381721368+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221152638e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317547659094e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332622604617e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440317+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214168734+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.67040239018765e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312317107+0j) [X1 X3] +
(3.60607186769027e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496493+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389555074+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309312072207e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.93324127702178e-07+0j) [Y1 X2 X3 Y4] +
(-0.0022939566113524615+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124239+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714589804297e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441852+0j) [Y1 X2 X4 Y5] +
(-8.091637199255312e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311865+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389678017138e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217887+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881552074+0j) [Y1 X2 X6 Y7] +
(0.005114473831660395+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464944055e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776294+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975433093e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381025+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630344+0j) [Y1 X2 X10 Y11] +
(-5.287660624737284e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744665312e-07+0j) [Y1 X2 X11 Y12] +
(-0.0045750076266392065+0j) [Y1 X2 X12 Y13] +
(1.93324127702178e-07+0j) [Y1 Y2 X3 X4] +
(0.0022939566113524615+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124239+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714589804297e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441852+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199255312e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311865+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389678017138e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217887+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881552074+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660395+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464944055e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776294+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975433093e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381025+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630344+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624737284e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744665312e-07+0j) [Y1 Y2 X11 X12] +
(-0.0045750076266392065+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475212381734e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939783+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128841006+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253793111647e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716555437628e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.1250703257977188+0j) [Y1 Z2 Y3] +
(-1.3807781480056397e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393084446776e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587407+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480056397e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393084446776e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587407+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897164+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781480056397e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986461817e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676629+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539175866676e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376507699993e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770608+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631569+0j) [Y1 Z2 Y3 Z5] +
(0.000594022154300555+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773244299948e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.000594022154300555+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244299948e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661952+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076848+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.07430598605403e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960949+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.974225379362598e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332103183108e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821402+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674751091+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132118+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599378271e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599378271e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018412782+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914614426e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574663824e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504313+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231581027+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797309+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125394+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024396+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.839420915415998e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538234+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0001940085702975701+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478339263e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446596391506e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696477+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369612+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565179445e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125394+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024396+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.839420915415998e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538234+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0001940085702975701+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478339263e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446596391506e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696477+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369612+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565179445e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.001028329237856253+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806609+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768791916487e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616076527e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798015+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616076527e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798015+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150951439295e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.002200964069500445+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.44459785405348e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.001172634831644192+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.839420915415998e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310132417822e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961075709e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.002261966062482338+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.002261966062482338+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453083016666e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619299+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126908+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742655817e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536651860109e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217039554e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487593+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641928493757e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832869+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.0001384017730355182+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422210716e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207150823e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423535+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487593+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641928493757e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832869+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.0001384017730355182+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422210716e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207150823e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423535+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287494106965e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.042743277013781514+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.001280306097349659+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182528+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425539141e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425539141e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.0052415353828038445+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282183883186e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943052637742e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839362+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907475+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914283+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336924+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413756078e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336924+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413756078e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219095+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219095+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002553+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245578+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046467+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009015319671e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476487193394e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.66134721335648e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015655+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658554652e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409991+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298335337e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230730671+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998845792574e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226877+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079230319246e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025528+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.0047672721882781165+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037469871e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339074+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.000715673424890913+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325318945607e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.291969486248552e-07+0j) [Y1 Z2 Z3 Y5] +
(0.001609531381721368+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221152638e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317547659094e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332622604617e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440317+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214168734+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.67040239018765e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317107+0j) [Y1 Y3] +
(3.60607186769027e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496493+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389555074+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309312072207e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111762+0j) [Z1] +
(-1.1908508082615868e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329057+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950634989+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603693100054e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508082615868e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329057+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950634989+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603693100054e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.25129445674591716+0j) [Z1 Z2] +
(-8.337746753703554e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273242+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214007+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735385402e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746753703554e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273242+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214007+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735385402e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830448+0j) [Z1 Z3] +
(-3.3440815563173056e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585305858737e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.3440815563173056e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585305858737e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.19936354537360815+0j) [Z1 Z4] +
(-3.099349243447143e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808795441035e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.099349243447143e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808795441035e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.1966177089034213+0j) [Z1 Z5] +
(0.05600733087780791+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851834053025e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780791+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851834053025e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314222+0j) [Z1 Z6] +
(0.05608468124661379+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.6522096695973055e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661379+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.6522096695973055e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2416466393601717+0j) [Z1 Z7] +
(0.27883454426723386+0j) [Z1 Z8] +
(0.27232518306605663+0j) [Z1 Z9] +
(-1.6148794138943045e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794138943045e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441763+0j) [Z1 Z10] +
(-2.1776646050417377e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646050417377e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364235+0j) [Z1 Z11] +
(0.2163103749863179+0j) [Z1 Z12] +
(0.21102659849791494+0j) [Z1 Z13] +
(-0.035839567953353434+0j) [X2 X3 Y4 Y5] +
(-2.1990516179003923e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.360956320265239e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831906+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516179003923e-07+0j) [X2 X3 X5 X6] +
(-2.360956320265239e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831906+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967284+0j) [X2 X3 Y6 Y7] +
(0.005368659358109643+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350639997204e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109643+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350639997204e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.036194123559042744+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457368+0j) [X2 X3 Y10 Y11] +
(2.1726691015006947e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691015006947e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976474+0j) [X2 X3 Y12 Y13] +
(0.035839567953353434+0j) [X2 Y3 Y4 X5] +
(2.1990516179003923e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.360956320265239e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831906+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516179003923e-07+0j) [X2 Y3 Y5 X6] +
(-2.360956320265239e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831906+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967284+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109643+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350639997204e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109643+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350639997204e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.036194123559042744+0j) [X2 Y3 Y8 X9] +
(0.025384657508457368+0j) [X2 Y3 Y10 X11] +
(-2.1726691015006947e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691015006947e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976474+0j) [X2 Y3 Y12 X13] +
(-3.887051672406717e-06+0j) [X2 Z3 X4] +
(-0.0051433917688251726+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962581+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706338465e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0051433917688251726+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962581+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706338465e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117602561e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489515306783e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908976+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780960921755e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112158825e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534391180592e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420191952406e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0192995605793638+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420191952406e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0192995605793638+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890099197097e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423776487463e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052994559175e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380182+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221704+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564320955164e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068887+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068887+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.80170750066689e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541845934444e-06+0j) [X2 Z3 X4 Z12] +
(8.81493730677566e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532435819384e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796816+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158474+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424490377687e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463112465007e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.01925750509525164+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676645623e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454826+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372207922e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.6430510685713724e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847182+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688708+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.2758831221822156e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424490377687e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463112465007e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.01925750509525164+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676645623e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454826+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372207922e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.6430510685713724e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847182+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688708+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.2758831221822156e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.1213327691104227+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791024048+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381545332719e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791024048+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381545332719e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802113+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826984+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646218+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288987802e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231089739495e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.000814531327095695+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184004162566e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184004162566e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130886+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499057+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901508+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447180013853e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819234+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226579+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.088250711313652e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429315037e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840056+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819234+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226579+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.088250711313652e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429315037e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840056+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.01460370472916208+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.874299071405513e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.01460370472916208+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.874299071405513e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022793+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946563387452e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946563387452e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354693055+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314684+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898838+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158457+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158457+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.775950527277471e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765761193365e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.1464963276705745e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.84620167133183e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.039359168022052984+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.9798257934902e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289091+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526722084688e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600798+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350502008564e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624845+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656541836e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907574+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942944+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560118159165e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334414+0j) [X2 Z3 Z5 X6] +
(-0.02873077955190553+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.9358677181543804e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.602116740589169e-06+0j) [X2 X4] +
(0.0004956762314914815+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831259+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348200645e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.035839567953353434+0j) [Y2 X3 X4 Y5] +
(2.1990516179003923e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.360956320265239e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831906+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516179003923e-07+0j) [Y2 X3 X5 Y6] +
(-2.360956320265239e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831906+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967284+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109643+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350639997204e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109643+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350639997204e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.036194123559042744+0j) [Y2 X3 X8 Y9] +
(0.025384657508457368+0j) [Y2 X3 X10 Y11] +
(-2.1726691015006947e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691015006947e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976474+0j) [Y2 X3 X12 Y13] +
(-0.035839567953353434+0j) [Y2 Y3 X4 X5] +
(-2.1990516179003923e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.360956320265239e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831906+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516179003923e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.360956320265239e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831906+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967284+0j) [Y2 Y3 X6 X7] +
(0.005368659358109643+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350639997204e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109643+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350639997204e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.036194123559042744+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457368+0j) [Y2 Y3 X10 X11] +
(2.1726691015006947e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691015006947e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976474+0j) [Y2 Y3 X12 X13] +
(1.6288532435819384e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796816+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158474+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051672406717e-06+0j) [Y2 Z3 Y4] +
(-0.0051433917688251726+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962581+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706338465e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0051433917688251726+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962581+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706338465e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117602561e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780960921755e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112158825e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489515306783e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908976+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534391180592e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420191952406e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0192995605793638+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420191952406e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0192995605793638+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890099197097e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423776487463e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052994559175e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221704+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380182+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564320955164e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068887+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068887+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.80170750066689e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541845934444e-06+0j) [Y2 Z3 Y4 Z12] +
(8.81493730677566e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424490377687e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463112465007e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.01925750509525164+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676645623e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454826+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372207922e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.6430510685713724e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847182+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688708+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.2758831221822156e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424490377687e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463112465007e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.01925750509525164+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676645623e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454826+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372207922e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.6430510685713724e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847182+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688708+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.2758831221822156e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447180013853e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.1213327691104227+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791024048+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381545332719e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791024048+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381545332719e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802113+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826984+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646218+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231089739495e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288987802e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.000814531327095695+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184004162566e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184004162566e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130886+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499057+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901508+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819234+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226579+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.088250711313652e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429315037e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840056+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819234+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226579+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.088250711313652e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429315037e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840056+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.01460370472916208+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.874299071405513e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.01460370472916208+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.874299071405513e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022793+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946563387452e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946563387452e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354693055+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314684+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898838+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158457+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158457+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.775950527277471e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765761193365e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.1464963276705745e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.84620167133183e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.039359168022052984+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.9798257934902e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289091+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526722084688e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600798+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350502008564e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624845+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656541836e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907574+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942944+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560118159165e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334414+0j) [Y2 Z3 Z5 Y6] +
(-0.02873077955190553+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.9358677181543804e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.602116740589169e-06+0j) [Y2 Y4] +
(0.0004956762314914815+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831259+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348200645e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831737+0j) [Z2] +
(1.602116740589169e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314914815+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831259+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348200645e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.602116740589169e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314914815+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831259+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348200645e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751405+0j) [Z2 Z3] +
(-9.509249751066774e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843147250282e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883830023+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751066774e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147250282e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883830023+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503233+0j) [Z2 Z4] +
(-1.1708301368967166e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467515521e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366193+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301368967166e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467515521e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366193+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838578+0j) [Z2 Z5] +
(0.019020423173040084+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156047556723e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173040084+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156047556723e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683246+0j) [Z2 Z6] +
(0.024389082531149728+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220983557e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149728+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220983557e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579972+0j) [Z2 Z7] +
(0.15071408121008306+0j) [Z2 Z8] +
(0.1869082047691258+0j) [Z2 Z9] +
(-1.0632283424601421e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283424601421e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468426+0j) [Z2 Z10] +
(1.1094407590405528e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407590405528e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314162+0j) [Z2 Z11] +
(0.14011289865354823+0j) [Z2 Z12] +
(0.15569010671752473+0j) [Z2 Z13] +
(0.005143391768825172+0j) [X3 X4 Y5 Y6] +
(0.009841749246962581+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706338465e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449037769e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676645623e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454826+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463112465007e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.01925750509525164+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372207922e-07+0j) [X3 X4 X8 X9] +
(-4.6430510685713724e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688708+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847182+0j) [X3 X4 Y11 Y12] +
(5.2758831221822156e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825172+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962581+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706338465e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449037769e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676645623e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454826+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463112465007e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.01925750509525164+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372207922e-07+0j) [X3 Y4 Y8 X9] +
(-4.6430510685713724e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688708+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847182+0j) [X3 Y4 Y11 X12] +
(5.2758831221822156e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051672406716e-06+0j) [X3 Z4 X5] +
(3.2118420191952406e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0192995605793638+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420191952406e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0192995605793638+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890099197097e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489515306783e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908976+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780960921755e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112158825e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534391180592e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052994559175e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423776487463e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068887+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068887+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.80170750066689e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380182+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221704+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564320955164e-06+0j) [X3 Z4 X5 Z11] +
(8.81493730677566e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541845934444e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532435819384e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796816+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158474+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791024048+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.686381545332719e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819232+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226579+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429315037e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.088250711313652e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840056+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791024048+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.686381545332719e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819232+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226579+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429315037e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.088250711313652e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840056+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042272+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646218+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826984+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184004162566e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184004162566e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130886+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288987802e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231089739495e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.000814531327095695+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901508+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499057+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447180013853e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.01460370472916208+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.874299071405513e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.01460370472916208+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.874299071405513e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946563387452e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158457+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946563387452e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158457+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702278+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898838+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314684+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.7759505272774704e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765761193365e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.84620167133183e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.02428211735469305+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.1464963276705745e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289091+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526722084688e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.039359168022052984+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.9798257934902e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624845+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656541836e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02599617759802113+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600798+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350502008564e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334414+0j) [X3 Z4 Z6 X7] +
(-0.02873077955190553+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.9358677181543804e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994117602561e-07+0j) [X3 X5] +
(0.0016638798784907574+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942944+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560118159165e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825172+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962581+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706338465e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449037769e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676645623e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454826+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463112465007e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.01925750509525164+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372207922e-07+0j) [Y3 X4 X8 Y9] +
(-4.6430510685713724e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688708+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847182+0j) [Y3 X4 X11 Y12] +
(5.2758831221822156e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825172+0j) [Y3 Y4 X5 X6] +
(0.009841749246962581+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706338465e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449037769e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676645623e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454826+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463112465007e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.01925750509525164+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372207922e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.6430510685713724e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688708+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847182+0j) [Y3 Y4 X11 X12] +
(5.2758831221822156e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532435819384e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796816+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158474+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051672406716e-06+0j) [Y3 Z4 Y5] +
(3.2118420191952406e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0192995605793638+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420191952406e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0192995605793638+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890099197097e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780960921755e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112158825e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489515306783e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908976+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534391180592e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052994559175e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423776487463e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068887+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068887+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.80170750066689e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221704+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380182+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564320955164e-06+0j) [Y3 Z4 Y5 Z11] +
(8.81493730677566e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541845934444e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791024048+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.686381545332719e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819232+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226579+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429315037e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.088250711313652e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840056+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791024048+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.686381545332719e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819232+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226579+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429315037e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.088250711313652e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840056+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447180013853e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042272+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646218+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826984+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184004162566e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184004162566e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130886+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231089739495e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288987802e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.000814531327095695+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901508+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499057+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.01460370472916208+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.874299071405513e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.01460370472916208+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.874299071405513e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946563387452e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158457+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946563387452e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158457+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702278+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898838+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314684+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.7759505272774704e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765761193365e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.84620167133183e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.02428211735469305+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.1464963276705745e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289091+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526722084688e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.039359168022052984+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.9798257934902e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624845+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656541836e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802113+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600798+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350502008564e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334414+0j) [Y3 Z4 Z6 Y7] +
(-0.02873077955190553+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.9358677181543804e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117602561e-07+0j) [Y3 Y5] +
(0.0016638798784907574+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942944+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560118159165e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831734+0j) [Z3] +
(-1.1708301368967166e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467515521e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366193+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301368967166e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467515521e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366193+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838578+0j) [Z3 Z4] +
(-9.509249751066774e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147250282e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883830023+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751066774e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147250282e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883830023+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503233+0j) [Z3 Z5] +
(0.024389082531149728+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220983557e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149728+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220983557e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579972+0j) [Z3 Z6] +
(0.019020423173040084+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156047556723e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173040084+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156047556723e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683246+0j) [Z3 Z7] +
(0.1869082047691258+0j) [Z3 Z8] +
(0.15071408121008306+0j) [Z3 Z9] +
(1.1094407590405528e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407590405528e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314162+0j) [Z3 Z10] +
(-1.0632283424601421e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283424601421e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468426+0j) [Z3 Z11] +
(0.15569010671752473+0j) [Z3 Z12] +
(0.14011289865354823+0j) [Z3 Z13] +
(-0.011982389010248019+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832996+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293594503162e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832996+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293594503162e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.00715693491985695+0j) [X4 X5 Y8 Y9] +
(-0.017680067952481494+0j) [X4 X5 Y10 Y11] +
(-3.6945132944908595e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132944908595e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480389+0j) [X4 X5 Y12 Y13] +
(0.011982389010248019+0j) [X4 Y5 Y6 X7] +
(0.007306759928832996+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293594503162e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832996+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293594503162e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.00715693491985695+0j) [X4 Y5 Y8 X9] +
(0.017680067952481494+0j) [X4 Y5 Y10 X11] +
(3.6945132944908595e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132944908595e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480389+0j) [X4 Y5 Y12 X13] +
(-1.2260484987974076e-05+0j) [X4 Z5 X6] +
(-1.2283337824714572e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756956825+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824714572e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756956825+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060857828919e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449080237354e-06+0j) [X4 Z5 X6 Z8] +
(-1.881850183094974e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921585+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730725+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.69239782850751e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997614052+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997614052+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913885010391e-06+0j) [X4 Z5 X6 Z11] +
(-4.5888551556709105e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694657+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750712385e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.9743117135179015e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840983+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535637+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218239125e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750712385e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.9743117135179015e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840983+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535637+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218239125e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731886707238e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561344+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731886707238e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561344+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928462398e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179538+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179538+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312895907207e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038759638e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775285382e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736525744e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736525744e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156156+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.02314513092952897+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847307+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026845+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864652014e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638308+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344675981291e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982174+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433269758e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.0395644163228932+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.5183622157412145e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719752+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765815151131e-07+0j) [X4 X6] +
(-4.253224225634494e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196013022+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010248019+0j) [Y4 X5 X6 Y7] +
(0.007306759928832996+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293594503162e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832996+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293594503162e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.00715693491985695+0j) [Y4 X5 X8 Y9] +
(0.017680067952481494+0j) [Y4 X5 X10 Y11] +
(3.6945132944908595e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132944908595e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480389+0j) [Y4 X5 X12 Y13] +
(-0.011982389010248019+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832996+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293594503162e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832996+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293594503162e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.00715693491985695+0j) [Y4 Y5 X8 X9] +
(-0.017680067952481494+0j) [Y4 Y5 X10 X11] +
(-3.6945132944908595e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132944908595e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480389+0j) [Y4 Y5 X12 X13] +
(0.008890731522694657+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484987974076e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824714572e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756956825+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824714572e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756956825+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060857828919e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449080237354e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.881850183094974e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730725+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921585+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.69239782850751e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997614052+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997614052+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913885010391e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.5888551556709105e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750712385e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.9743117135179015e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840983+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535637+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218239125e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750712385e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.9743117135179015e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840983+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535637+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218239125e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731886707238e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561344+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731886707238e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561344+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928462398e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179538+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179538+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312895907207e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038759638e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775285382e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736525744e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736525744e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156156+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.02314513092952897+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847307+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026845+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864652014e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638308+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344675981291e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982174+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433269758e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.0395644163228932+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.5183622157412145e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719752+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765815151131e-07+0j) [Y4 Y6] +
(-4.253224225634494e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196013022+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145623+0j) [Z4] +
(-5.92976581515113e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225634494e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196013022+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.92976581515113e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225634494e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196013022+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985664+0j) [Z4 Z5] +
(0.018266834869375654+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174771687336e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375654+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174771687336e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1370119167404076+0j) [Z4 Z6] +
(0.010960074940542656+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468366190495e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542656+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468366190495e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1489943057506556+0j) [Z4 Z7] +
(0.14960702684445296+0j) [Z4 Z8] +
(0.15676396176430993+0j) [Z4 Z9] +
(1.8782101247815585e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247815585e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237612+0j) [Z4 Z10] +
(-1.8163031697093014e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031697093014e-06+0j) [Z4 Y11 Z12 Y13] +
(0.1425799771248576+0j) [Z4 Z11] +
(0.11383573679388659+0j) [Z4 Z12] +
(0.1521504070886905+0j) [Z4 Z13] +
(1.2283337824714572e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756956825+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750712385e-07+0j) [X5 X6 X8 X9] +
(5.9743117135179015e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535637+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840983+0j) [X5 X6 Y11 Y12] +
(-4.556569218239125e-06+0j) [X5 X6 X12 X13] +
(-1.2283337824714572e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756956825+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750712385e-07+0j) [X5 Y6 Y8 X9] +
(5.9743117135179015e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535637+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840983+0j) [X5 Y6 Y11 X12] +
(-4.556569218239125e-06+0j) [X5 Y6 Y12 X13] +
(-1.226048498797408e-05+0j) [X5 Z6 X7] +
(-1.881850183094974e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449080237354e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997614052+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997614052+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913885010391e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921585+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730725+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.69239782850751e-06+0j) [X5 Z6 X7 Z11] +
(-4.5888551556709105e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694657+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731886707238e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561344+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731886707238e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561344+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179538+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736525744e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179538+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736525744e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928462398e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775285382e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038759638e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156156+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02314513092952897+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026845+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312895907207e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847307+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344675981291e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982174+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864652014e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638308+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.5183622157412145e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719752+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.854060857828919e-06+0j) [X5 X7] +
(-6.290028433269758e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.0395644163228932+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824714572e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756956825+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750712385e-07+0j) [Y5 X6 X8 Y9] +
(5.9743117135179015e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535637+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840983+0j) [Y5 X6 X11 Y12] +
(-4.556569218239125e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337824714572e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756956825+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750712385e-07+0j) [Y5 Y6 Y8 Y9] +
(5.9743117135179015e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535637+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840983+0j) [Y5 Y6 X11 X12] +
(-4.556569218239125e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694657+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.226048498797408e-05+0j) [Y5 Z6 Y7] +
(-1.881850183094974e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449080237354e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997614052+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997614052+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913885010391e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730725+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921585+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.69239782850751e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.5888551556709105e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731886707238e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561344+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731886707238e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561344+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179538+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736525744e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179538+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736525744e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928462398e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775285382e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038759638e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156156+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02314513092952897+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026845+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312895907207e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847307+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344675981291e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982174+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864652014e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638308+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.5183622157412145e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719752+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060857828919e-06+0j) [Y5 Y7] +
(-6.290028433269758e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.0395644163228932+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145625+0j) [Z5] +
(0.010960074940542656+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468366190495e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542656+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468366190495e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1489943057506556+0j) [Z5 Z6] +
(0.018266834869375654+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174771687336e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375654+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174771687336e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1370119167404076+0j) [Z5 Z7] +
(0.15676396176430993+0j) [Z5 Z8] +
(0.14960702684445296+0j) [Z5 Z9] +
(-1.8163031697093014e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031697093014e-06+0j) [Z5 Y10 Z11 Y12] +
(0.1425799771248576+0j) [Z5 Z10] +
(1.8782101247815585e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247815585e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237612+0j) [Z5 Z11] +
(0.1521504070886905+0j) [Z5 Z12] +
(0.11383573679388659+0j) [Z5 Z13] +
(-0.013873381748426058+0j) [X6 X7 Y8 Y9] +
(-0.01782514099578665+0j) [X6 X7 Y10 Y11] +
(-1.035847760160686e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.035847760160686e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651444+0j) [X6 X7 Y12 Y13] +
(0.013873381748426058+0j) [X6 Y7 Y8 X9] +
(0.01782514099578665+0j) [X6 Y7 Y10 X11] +
(1.035847760160686e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.035847760160686e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651444+0j) [X6 Y7 Y12 X13] +
(0.00029219862611100297+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393505859796e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611100297+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393505859796e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564919002+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500243673e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500243673e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848248+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.02510495713884461+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671552+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173058+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173058+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860070606135e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559443869e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.52437384873396e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228348490286e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345948+0j) [X6 Z7 Z8 X10] +
(-3.277483195593297e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456948+0j) [X6 Z7 Z9 X10] +
(-3.6102971306518956e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389144043+0j) [X6 Z8 Z9 X10] +
(-3.7696594520638633e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426058+0j) [Y6 X7 X8 Y9] +
(0.01782514099578665+0j) [Y6 X7 X10 Y11] +
(1.035847760160686e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.035847760160686e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651444+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426058+0j) [Y6 Y7 X8 X9] +
(-0.01782514099578665+0j) [Y6 Y7 X10 X11] +
(-1.035847760160686e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.035847760160686e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651444+0j) [Y6 Y7 X12 X13] +
(0.00029219862611100297+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393505859796e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611100297+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393505859796e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564919002+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500243673e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500243673e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848248+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.02510495713884461+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671552+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173058+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173058+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860070606135e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559443869e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.52437384873396e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228348490286e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345948+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195593297e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456948+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971306518956e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389144043+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594520638633e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615412+0j) [Z6] +
(0.030787505389144043+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594520638633e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389144043+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594520638633e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270188+0j) [Z6 Z7] +
(0.1675665326546127+0j) [Z6 Z8] +
(0.18143991440303875+0j) [Z6 Z9] +
(-1.8551201215669972e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201215669972e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682685+0j) [Z6 Z10] +
(-2.8909678817276832e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678817276832e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261352+0j) [Z6 Z11] +
(0.1340171526196371+0j) [Z6 Z12] +
(0.15138327161428855+0j) [Z6 Z13] +
(-0.00029219862611100297+0j) [X7 X8 Y9 Y10] +
(3.3281393505859796e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611100297+0j) [X7 Y8 Y9 X10] +
(-3.3281393505859796e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.313145500243673e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173058+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.313145500243673e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173058+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564919002+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671552+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.02510495713884461+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086007060614e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559443869e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228348490286e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848248+0j) [X7 Z8 Z9 X11] +
(-6.52437384873396e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456948+0j) [X7 Z8 Z10 X11] +
(-3.6102971306518956e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345948+0j) [X7 Z9 Z10 X11] +
(-3.277483195593297e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611100297+0j) [Y7 X8 X9 Y10] +
(-3.3281393505859796e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611100297+0j) [Y7 Y8 X9 X10] +
(3.3281393505859796e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.313145500243673e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173058+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.313145500243673e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173058+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564919002+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671552+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.02510495713884461+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086007060614e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559443869e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228348490286e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848248+0j) [Y7 Z8 Z9 Y11] +
(-6.52437384873396e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456948+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971306518956e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345948+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195593297e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615414+0j) [Z7] +
(0.18143991440303875+0j) [Z7 Z8] +
(0.1675665326546127+0j) [Z7 Z9] +
(-2.8909678817276832e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678817276832e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261352+0j) [Z7 Z10] +
(-1.8551201215669972e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201215669972e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682685+0j) [Z7 Z11] +
(0.15138327161428855+0j) [Z7 Z12] +
(0.1340171526196371+0j) [Z7 Z13] +
(-0.009560705729135921+0j) [X8 X9 Y10 Y11] +
(6.628614201766583e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201766583e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561852+0j) [X8 X9 Y12 Y13] +
(0.009560705729135921+0j) [X8 Y9 Y10 X11] +
(-6.628614201766583e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201766583e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561852+0j) [X8 Y9 Y12 X13] +
(0.009560705729135921+0j) [Y8 X9 X10 Y11] +
(-6.628614201766583e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201766583e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561852+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135921+0j) [Y8 Y9 X10 X11] +
(6.628614201766583e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201766583e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561852+0j) [Y8 Y9 X12 X13] +
(1.3693525634718176+0j) [Z8] +
(-1.597317197833026e-06+0j) [Z8 X10 Z11 X12] +
(-1.597317197833026e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852588+0j) [Z8 Z10] +
(-9.344557776563677e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557776563677e-07+0j) [Z8 Y11 Z12 Y13] +
(0.1472294321876618+0j) [Z8 Z11] +
(0.14973486803496927+0j) [Z8 Z12] +
(-9.344557776563677e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557776563677e-07+0j) [Z9 Y10 Z11 Y12] +
(0.1472294321876618+0j) [Z9 Z10] +
(-1.597317197833026e-06+0j) [Z9 X11 Z12 X13] +
(-1.597317197833026e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852588+0j) [Z9 Z11] +
(0.14973486803496927+0j) [Z9 Z13] +
(-0.028685183716105907+0j) [X10 X11 Y12 Y13] +
(0.028685183716105907+0j) [X10 Y11 Y12 X13] +
(-1.0722312157697747e-05+0j) [X10 Z11 X12] +
(7.95441317637289e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372454784e-06+0j) [X10 X12] +
(0.028685183716105907+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105907+0j) [Y10 Y11 X12 X13] +
(-1.0722312157697747e-05+0j) [Y10 Z11 Y12] +
(7.95441317637289e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372454784e-06+0j) [Y10 Y12] +
(-8.194261372454784e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372454784e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388914+0j) [Z10 Z11] +
(0.11270386920332226+0j) [Z10 Z12] +
(0.14138905291942816+0j) [Z10 Z13] +
(-1.0722312157697745e-05+0j) [X11 Z12 X13] +
(7.954413176372892e-06+0j) [X11 X13] +
(-1.0722312157697745e-05+0j) [Y11 Z12 Y13] +
(7.954413176372892e-06+0j) [Y11 Y13] +
(0.14138905291942816+0j) [Z11 Z12] +
(0.11270386920332226+0j) [Z11 Z13] +
(0.8084581961720478+0j) [Z12] +
(0.15435748657223636+0j) [Z12 Z13] +
(0.8084581961720486+0j) [Z13]
  (-46.463906788689) [I0]
+ (0.7829661725950189) [Z10]
+ (0.782966172595019) [Z11]
+ (0.8084581961720463) [Z12]
+ (0.8084581961720467) [Z13]
+ (1.2034402289145643) [Z4]
+ (1.2034402289145654) [Z5]
+ (1.3096862988615428) [Z6]
+ (1.3096862988615432) [Z7]
+ (1.3693525634718182) [Z8]
+ (1.3693525634718182) [Z9]
+ (1.6538942226831659) [Z2]
+ (1.6538942226831668) [Z3]
+ (12.412630742111787) [Z0]
+ (12.412630742111787) [Z1]
+ (-8.194261372016918e-06) [Y10 Y12]
+ (-8.194261372016918e-06) [X10 X12]
+ (-1.8540608580147535e-06) [Y5 Y7]
+ (-1.8540608580147535e-06) [X5 X7]
+ (-7.764994118537553e-07) [Y3 Y5]
+ (-7.764994118537553e-07) [X3 X5]
+ (-5.929765815976579e-07) [Y4 Y6]
+ (-5.929765815976579e-07) [X4 X6]
+ (1.6021167406623449e-06) [Y2 Y4]
+ (1.6021167406623449e-06) [X2 X4]
+ (7.954413176413522e-06) [Y11 Y13]
+ (7.954413176413522e-06) [X11 X13]
+ (0.0032769719312315815) [Y1 Y3]
+ (0.0032769719312315815) [X1 X3]
+ (0.10433064780651388) [Y0 Y2]
+ (0.10433064780651388) [X0 X2]
+ (0.11270386920332223) [Z10 Z12]
+ (0.11270386920332223) [Z11 Z13]
+ (0.11383573679388662) [Z4 Z12]
+ (0.11383573679388662) [Z5 Z13]
+ (0.11952438964682693) [Z6 Z10]
+ (0.11952438964682693) [Z7 Z11]
+ (0.12489990917237616) [Z4 Z10]
+ (0.12489990917237616) [Z5 Z11]
+ (0.12495807739503198) [Z2 Z4]
+ (0.12495807739503198) [Z3 Z5]
+ (0.12799502492468384) [Z2 Z10]
+ (0.12799502492468384) [Z3 Z11]
+ (0.13401715261963715) [Z6 Z12]
+ (0.13401715261963715) [Z7 Z13]
+ (0.13701191674040777) [Z4 Z6]
+ (0.13701191674040777) [Z5 Z7]
+ (0.1373495306426134) [Z6 Z11]
+ (0.1373495306426134) [Z7 Z10]
+ (0.1373910476268321) [Z2 Z6]
+ (0.1373910476268321) [Z3 Z7]
+ (0.1376687264585259) [Z8 Z10]
+ (0.1376687264585259) [Z9 Z11]
+ (0.1401128986535478) [Z2 Z12]
+ (0.1401128986535478) [Z3 Z13]
+ (0.1413890529194281) [Z10 Z13]
+ (0.1413890529194281) [Z11 Z12]
+ (0.14257997712485765) [Z4 Z11]
+ (0.14257997712485765) [Z5 Z10]
+ (0.14722943218766188) [Z8 Z11]
+ (0.14722943218766188) [Z9 Z10]
+ (0.14899430575065575) [Z4 Z7]
+ (0.14899430575065575) [Z5 Z6]
+ (0.1492635514738891) [Z10 Z11]
+ (0.14960702684445315) [Z4 Z8]
+ (0.14960702684445315) [Z5 Z9]
+ (0.14973486803496933) [Z8 Z12]
+ (0.14973486803496933) [Z9 Z13]
+ (0.15071408121008267) [Z2 Z8]
+ (0.15071408121008267) [Z3 Z9]
+ (0.15138327161428855) [Z6 Z13]
+ (0.15138327161428855) [Z7 Z12]
+ (0.15215040708869054) [Z4 Z13]
+ (0.15215040708869054) [Z5 Z12]
+ (0.15337968243314126) [Z2 Z11]
+ (0.15337968243314126) [Z3 Z10]
+ (0.15435748657223625) [Z12 Z13]
+ (0.15569010671752423) [Z2 Z13]
+ (0.15569010671752423) [Z3 Z12]
+ (0.15582269051553116) [Z8 Z13]
+ (0.15582269051553116) [Z9 Z12]
+ (0.15676396176431012) [Z4 Z9]
+ (0.15676396176431012) [Z5 Z8]
+ (0.15755314797985684) [Z4 Z5]
+ (0.16079764534838542) [Z2 Z5]
+ (0.16079764534838542) [Z3 Z4]
+ (0.16756653265461294) [Z6 Z8]
+ (0.16756653265461294) [Z7 Z9]
+ (0.1685348656157992) [Z2 Z7]
+ (0.1685348656157992) [Z3 Z6]
+ (0.18143991440303905) [Z6 Z9]
+ (0.18143991440303905) [Z7 Z8]
+ (0.18189085790751286) [Z2 Z3]
+ (0.18690820476912517) [Z2 Z9]
+ (0.18690820476912517) [Z3 Z8]
+ (0.19299723935364282) [Z0 Z10]
+ (0.19299723935364282) [Z1 Z11]
+ (0.19392534613270243) [Z6 Z7]
+ (0.19661770890342192) [Z0 Z4]
+ (0.19661770890342192) [Z1 Z5]
+ (0.19936354537360873) [Z0 Z5]
+ (0.19936354537360873) [Z1 Z4]
+ (0.20072866460441813) [Z0 Z11]
+ (0.20072866460441813) [Z1 Z10]
+ (0.21102659849791533) [Z0 Z12]
+ (0.21102659849791533) [Z1 Z13]
+ (0.21631037498631828) [Z0 Z13]
+ (0.21631037498631828) [Z1 Z12]
+ (0.22003977334376112) [Z8 Z9]
+ (0.23671080783830395) [Z0 Z2]
+ (0.23671080783830395) [Z1 Z3]
+ (0.24164663936017258) [Z0 Z6]
+ (0.24164663936017258) [Z1 Z7]
+ (0.24853483371314317) [Z0 Z7]
+ (0.24853483371314317) [Z1 Z6]
+ (0.25129445674591655) [Z0 Z3]
+ (0.25129445674591655) [Z1 Z2]
+ (0.27232518306605724) [Z0 Z8]
+ (0.27232518306605724) [Z1 Z9]
+ (0.27883454426723453) [Z0 Z9]
+ (0.27883454426723453) [Z1 Z8]
+ (1.1861763734860524) [Z0 Z1]
+ (-1.22604849891371e-05) [Y4 Z5 Y6]
+ (-1.22604849891371e-05) [X4 Z5 X6]
+ (-1.2260484989137097e-05) [Y5 Z6 Y7]
+ (-1.2260484989137097e-05) [X5 Z6 X7]
+ (-1.0722312156941663e-05) [Y11 Z12 Y13]
+ (-1.0722312156941663e-05) [X11 Z12 X13]
+ (-1.0722312156941661e-05) [Y10 Z11 Y12]
+ (-1.0722312156941661e-05) [X10 Z11 X12]
+ (-3.887051673397606e-06) [Y3 Z4 Y5]
+ (-3.887051673397606e-06) [X3 Z4 X5]
+ (-3.887051673397604e-06) [Y2 Z3 Y4]
+ (-3.887051673397604e-06) [X2 Z3 X4]
+ (0.12507032579771793) [Y0 Z1 Y2]
+ (0.12507032579771793) [X0 Z1 X2]
+ (0.12507032579771798) [Y1 Z2 Y3]
+ (0.12507032579771798) [X1 Z2 X3]
+ (-0.038314670294803906) [Y4 Y5 X12 X13]
+ (-0.038314670294803906) [X4 X5 Y12 Y13]
+ (-0.03619412355904252) [Y2 Y3 X8 X9]
+ (-0.03619412355904252) [X2 X3 Y8 Y9]
+ (-0.03583956795335347) [Y2 Y3 X4 X5]
+ (-0.03583956795335347) [X2 X3 Y4 Y5]
+ (-0.031143817988967114) [Y2 Y3 X6 X7]
+ (-0.031143817988967114) [X2 X3 Y6 Y7]
+ (-0.028685183716105903) [Y10 Y11 X12 X13]
+ (-0.028685183716105903) [X10 X11 Y12 Y13]
+ (-0.025996177598021208) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021208) [X3 Z4 Z5 X7]
+ (-0.025384657508457396) [Y2 Y3 X10 X11]
+ (-0.025384657508457396) [X2 X3 Y10 Y11]
+ (-0.01902824244384731) [Y3 Y4 X11 X12]
+ (-0.01902824244384731) [X3 X4 Y11 Y12]
+ (-0.017825140995786474) [Y6 Y7 X10 X11]
+ (-0.017825140995786474) [X6 X7 Y10 Y11]
+ (-0.017680067952481466) [Y4 Y5 X10 X11]
+ (-0.017680067952481466) [X4 X5 Y10 Y11]
+ (-0.017366118994651417) [Y6 Y7 X12 X13]
+ (-0.017366118994651417) [X6 X7 Y12 Y13]
+ (-0.01557720806397642) [Y2 Y3 X12 X13]
+ (-0.01557720806397642) [X2 X3 Y12 Y13]
+ (-0.014583648907612613) [Y0 Y1 X2 X3]
+ (-0.014583648907612613) [X0 X1 Y2 Y3]
+ (-0.013873381748426127) [Y6 Y7 X8 X9]
+ (-0.013873381748426127) [X6 X7 Y8 Y9]
+ (-0.011982389010247972) [Y4 Y5 X6 X7]
+ (-0.011982389010247972) [X4 X5 Y6 Y7]
+ (-0.009560705729135982) [Y8 Y9 X10 X11]
+ (-0.009560705729135982) [X8 X9 Y10 Y11]
+ (-0.007731425250775323) [Y0 Y1 X10 X11]
+ (-0.007731425250775323) [X0 X1 Y10 Y11]
+ (-0.007156934919856959) [Y4 Y5 X8 X9]
+ (-0.007156934919856959) [X4 X5 Y8 Y9]
+ (-0.0068881943529705844) [Y0 Y1 X6 X7]
+ (-0.0068881943529705844) [X0 X1 Y6 Y7]
+ (-0.006509361201177252) [Y0 Y1 X8 X9]
+ (-0.006509361201177252) [X0 X1 Y8 Y9]
+ (-0.006087822480561854) [Y8 Y9 X12 X13]
+ (-0.006087822480561854) [X8 X9 Y12 Y13]
+ (-0.005283776488402961) [Y0 Y1 X12 X13]
+ (-0.005283776488402961) [X0 X1 Y12 Y13]
+ (-0.005143391768825096) [Y3 X4 X5 Y6]
+ (-0.005143391768825096) [X3 Y4 Y5 X6]
+ (-0.004684903388155211) [Y1 X2 X6 Y7]
+ (-0.004684903388155211) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155211) [X1 X2 X6 X7]
+ (-0.004684903388155211) [X1 Y2 Y6 X7]
+ (-0.004575007626639194) [Y1 X2 X12 Y13]
+ (-0.004575007626639194) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639194) [X1 X2 X12 X13]
+ (-0.004575007626639194) [X1 Y2 Y12 X13]
+ (-0.004424855449441864) [Y1 X2 X4 Y5]
+ (-0.004424855449441864) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441864) [X1 X2 X4 X5]
+ (-0.004424855449441864) [X1 Y2 Y4 X5]
+ (-0.0034795118903343295) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343295) [X2 Z3 Z5 X6]
+ (-0.0034795118903343295) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343295) [X3 Z4 Z6 X7]
+ (-0.0027458364701868233) [Y0 Y1 X4 X5]
+ (-0.0027458364701868233) [X0 X1 Y4 Y5]
+ (-0.0017992194936630166) [Y1 X2 X10 Y11]
+ (-0.0017992194936630166) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630166) [X1 X2 X10 X11]
+ (-0.0017992194936630166) [X1 Y2 Y10 X11]
+ (-0.00029219862611106997) [Y7 Y8 X9 X10]
+ (-0.00029219862611106997) [X7 X8 Y9 Y10]
+ (-8.194261372016918e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372016918e-06) [Z10 X11 Z12 X13]
+ (-7.801707500415896e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500415896e-06) [X2 Z3 X4 Z11]
+ (-7.801707500415896e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500415896e-06) [X3 Z4 X5 Z10]
+ (-4.643051068429501e-06) [Y3 X4 X10 Y11]
+ (-4.643051068429501e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068429501e-06) [X3 X4 X10 X11]
+ (-4.643051068429501e-06) [X3 Y4 Y10 X11]
+ (-4.5888551556881044e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551556881044e-06) [X4 Z5 X6 Z13]
+ (-4.5888551556881044e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551556881044e-06) [X5 Z6 X7 Z12]
+ (-4.556569218090887e-06) [Y5 X6 X12 Y13]
+ (-4.556569218090887e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218090887e-06) [X5 X6 X12 X13]
+ (-4.556569218090887e-06) [X5 Y6 Y12 X13]
+ (-3.694513294406705e-06) [Y4 X5 X11 Y12]
+ (-3.694513294406705e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294406705e-06) [X4 X5 X11 X12]
+ (-3.694513294406705e-06) [X4 Y5 Y11 X12]
+ (-3.3440815566094088e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815566094088e-06) [Z0 X5 Z6 X7]
+ (-3.3440815566094088e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815566094088e-06) [Z1 X4 Z5 X6]
+ (-3.1586564319863944e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564319863944e-06) [X2 Z3 X4 Z10]
+ (-3.1586564319863944e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564319863944e-06) [X3 Z4 X5 Z11]
+ (-3.0993492437201903e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492437201903e-06) [Z0 X4 Z5 X6]
+ (-3.0993492437201903e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492437201903e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816020623e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816020623e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816020623e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816020623e-06) [Z7 X10 Z11 X12]
+ (-2.177664604868553e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604868553e-06) [Z0 X10 Z11 X12]
+ (-2.177664604868553e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604868553e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832716875e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832716875e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832716875e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832716875e-06) [X5 Z6 X7 Z8]
+ (-1.8551201213984099e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201213984099e-06) [Z6 X10 Z11 X12]
+ (-1.8551201213984099e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201213984099e-06) [Z7 X11 Z12 X13]
+ (-1.8540608580147535e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608580147535e-06) [X4 Z5 X6 Z7]
+ (-1.816303169582024e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169582024e-06) [Z4 X11 Z12 X13]
+ (-1.816303169582024e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169582024e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285762533e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285762533e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285762533e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285762533e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137169022e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137169022e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137169022e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137169022e-06) [Z1 X10 Z11 X12]
+ (-1.597317197692848e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197692848e-06) [Z8 X10 Z11 X12]
+ (-1.597317197692848e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197692848e-06) [Z9 X11 Z12 X13]
+ (-1.4548424491257585e-06) [Y3 X4 X6 Y7]
+ (-1.4548424491257585e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424491257585e-06) [X3 X4 X6 X7]
+ (-1.4548424491257585e-06) [X3 Y4 Y6 X7]
+ (-1.39804490816631e-06) [Y4 Z5 Y6 Z8]
+ (-1.39804490816631e-06) [X4 Z5 X6 Z8]
+ (-1.39804490816631e-06) [Y5 Z6 Y7 Z9]
+ (-1.39804490816631e-06) [X5 Z6 X7 Z9]
+ (-1.1954890100266997e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890100266997e-06) [X2 Z3 X4 Z7]
+ (-1.1954890100266997e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890100266997e-06) [X3 Z4 X5 Z6]
+ (-1.1908508084180548e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508084180548e-06) [Z0 X3 Z4 X5]
+ (-1.1908508084180548e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508084180548e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370504406e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370504406e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370504406e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370504406e-06) [Z3 X4 Z5 X6]
+ (-1.0632283422133387e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283422133387e-06) [Z2 X10 Z11 X12]
+ (-1.0632283422133387e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283422133387e-06) [Z3 X11 Z12 X13]
+ (-1.0358477602036525e-06) [Y6 X7 X11 Y12]
+ (-1.0358477602036525e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477602036525e-06) [X6 X7 X11 X12]
+ (-1.0358477602036525e-06) [X6 Y7 Y11 X12]
+ (-9.509249751800896e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751800896e-07) [Z2 X4 Z5 X6]
+ (-9.509249751800896e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751800896e-07) [Z3 X5 Z6 X7]
+ (-9.34455777518027e-07) [Z8 Y11 Z12 Y13]
+ (-9.34455777518027e-07) [Z8 X11 Z12 X13]
+ (-9.34455777518027e-07) [Z9 Y10 Z11 Y12]
+ (-9.34455777518027e-07) [Z9 X10 Z11 X12]
+ (-8.337746754967356e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754967356e-07) [Z0 X2 Z3 X4]
+ (-8.337746754967356e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754967356e-07) [Z1 X3 Z4 X5]
+ (-7.956895372802302e-07) [Y3 X4 X8 Y9]
+ (-7.956895372802302e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372802302e-07) [X3 X4 X8 X9]
+ (-7.956895372802302e-07) [X3 Y4 Y8 X9]
+ (-7.764994118537554e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118537554e-07) [X2 Z3 X4 Z5]
+ (-5.929765815976579e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815976579e-07) [Z4 X5 Z6 X7]
+ (-5.770052995315241e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995315241e-07) [X2 Z3 X4 Z9]
+ (-5.770052995315241e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995315241e-07) [X3 Z4 X5 Z8]
+ (-5.471647744657892e-07) [Y1 Y2 X11 X12]
+ (-5.471647744657892e-07) [X1 X2 Y11 Y12]
+ (-4.838052751053777e-07) [Y5 X6 X8 Y9]
+ (-4.838052751053777e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751053777e-07) [X5 X6 X8 X9]
+ (-4.838052751053777e-07) [X5 Y6 Y8 X9]
+ (-3.570761329213192e-07) [Y0 X1 X3 Y4]
+ (-3.570761329213192e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329213192e-07) [X0 X1 X3 X4]
+ (-3.570761329213192e-07) [X0 Y1 Y3 X4]
+ (-2.447323128892185e-07) [Y0 X1 X5 Y6]
+ (-2.447323128892185e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128892185e-07) [X0 X1 X5 X6]
+ (-2.447323128892185e-07) [X0 Y1 Y5 X6]
+ (-2.1990516187035106e-07) [Y2 X3 X5 Y6]
+ (-2.1990516187035106e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516187035106e-07) [X2 X3 X5 X6]
+ (-2.1990516187035106e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771199162e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771199162e-07) [X1 Y2 Y3 X4]
+ (-1.291969486347455e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486347455e-07) [X1 Z2 Z3 X5]
+ (1.7379332623791723e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623791723e-07) [X0 Z1 Z3 X4]
+ (1.7379332623791723e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623791723e-07) [X1 Z2 Z4 X5]
+ (1.9332412771199162e-07) [Y1 Y2 X3 X4]
+ (1.9332412771199162e-07) [X1 X2 Y3 Y4]
+ (2.1868423774870615e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423774870615e-07) [X2 Z3 X4 Z8]
+ (2.1868423774870615e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423774870615e-07) [X3 Z4 X5 Z9]
+ (2.593534390990589e-07) [Y2 Z3 Y4 Z6]
+ (2.593534390990589e-07) [X2 Z3 X4 Z6]
+ (2.593534390990589e-07) [Y3 Z4 Y5 Z7]
+ (2.593534390990589e-07) [X3 Z4 X5 Z7]
+ (3.6060718679182504e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718679182504e-07) [X0 Z1 Z2 X4]
+ (3.6060718679182504e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718679182504e-07) [X1 Z3 Z4 X5]
+ (5.471647744657892e-07) [Y1 X2 X11 Y12]
+ (5.471647744657892e-07) [X1 Y2 Y11 X12]
+ (5.627851911516509e-07) [Y0 X1 X11 Y12]
+ (5.627851911516509e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911516509e-07) [X0 X1 X11 X12]
+ (5.627851911516509e-07) [X0 Y1 Y11 X12]
+ (6.628614201748209e-07) [Y8 X9 X11 Y12]
+ (6.628614201748209e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201748209e-07) [X8 X9 X11 X12]
+ (6.628614201748209e-07) [X8 Y9 Y11 X12]
+ (1.1094407593152852e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407593152852e-06) [Z2 X11 Z12 X13]
+ (1.1094407593152852e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407593152852e-06) [Z3 X10 Z11 X12]
+ (1.6021167406623449e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406623449e-06) [Z2 X3 Z4 X5]
+ (1.8782101248246814e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101248246814e-06) [Z4 X10 Z11 X12]
+ (1.8782101248246814e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101248246814e-06) [Z5 X11 Z12 X13]
+ (2.172669101528624e-06) [Y2 X3 X11 Y12]
+ (2.172669101528624e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101528624e-06) [X2 X3 X11 X12]
+ (2.172669101528624e-06) [X2 Y3 Y11 X12]
+ (3.1174479461549258e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479461549258e-06) [X0 Z2 Z3 X4]
+ (3.5390541845289835e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541845289835e-06) [X2 Z3 X4 Z12]
+ (3.5390541845289835e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541845289835e-06) [X3 Z4 X5 Z13]
+ (4.281913884834788e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884834788e-06) [X4 Z5 X6 Z11]
+ (4.281913884834788e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884834788e-06) [X5 Z6 X7 Z10]
+ (5.275883122158967e-06) [Y3 X4 X12 Y13]
+ (5.275883122158967e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122158967e-06) [X3 X4 X12 X13]
+ (5.275883122158967e-06) [X3 Y4 Y12 X13]
+ (5.97431171341104e-06) [Y5 X6 X10 Y11]
+ (5.97431171341104e-06) [Y5 Y6 Y10 Y11]
+ (5.97431171341104e-06) [X5 X6 X10 X11]
+ (5.97431171341104e-06) [X5 Y6 Y10 X11]
+ (7.954413176413522e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176413522e-06) [X10 Z11 X12 Z13]
+ (8.814937306687949e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306687949e-06) [X2 Z3 X4 Z13]
+ (8.814937306687949e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306687949e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611106997) [Y7 X8 X9 Y10]
+ (0.00029219862611106997) [X7 Y8 Y9 X10]
+ (0.0004956762314916663) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916663) [X2 Z4 Z5 X6]
+ (0.0011059037691896398) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896398) [X0 Z1 X2 Z5]
+ (0.0011059037691896398) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896398) [X1 Z2 X3 Z4]
+ (0.0016638798784907672) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907672) [X2 Z3 Z4 X6]
+ (0.0016638798784907672) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907672) [X3 Z5 Z6 X7]
+ (0.0017560707018411982) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411982) [X0 Z1 X2 Z11]
+ (0.0017560707018411982) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411982) [X1 Z2 X3 Z10]
+ (0.002326230623158023) [Y0 Z1 Y2 Z13]
+ (0.002326230623158023) [X0 Z1 X2 Z13]
+ (0.002326230623158023) [Y1 Z2 Y3 Z12]
+ (0.002326230623158023) [X1 Z2 X3 Z12]
+ (0.0027458364701868233) [Y0 X1 X4 Y5]
+ (0.0027458364701868233) [X0 Y1 Y4 X5]
+ (0.002929768674750982) [Y0 Z1 Y2 Z9]
+ (0.002929768674750982) [X0 Z1 X2 Z9]
+ (0.002929768674750982) [Y1 Z2 Y3 Z8]
+ (0.002929768674750982) [X1 Z2 X3 Z8]
+ (0.003276971931231582) [Y0 Z1 Y2 Z3]
+ (0.003276971931231582) [X0 Z1 X2 Z3]
+ (0.0033476175306661254) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661254) [X0 Z1 X2 Z7]
+ (0.0033476175306661254) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661254) [X1 Z2 X3 Z6]
+ (0.0035552901955042153) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042153) [X0 Z1 X2 Z10]
+ (0.0035552901955042153) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042153) [X1 Z2 X3 Z11]
+ (0.005143391768825096) [Y3 Y4 X5 X6]
+ (0.005143391768825096) [X3 X4 Y5 Y6]
+ (0.005283776488402961) [Y0 X1 X12 Y13]
+ (0.005283776488402961) [X0 Y1 Y12 X13]
+ (0.005530759218631504) [Y0 Z1 Y2 Z4]
+ (0.005530759218631504) [X0 Z1 X2 Z4]
+ (0.005530759218631504) [Y1 Z2 Y3 Z5]
+ (0.005530759218631504) [X1 Z2 X3 Z5]
+ (0.006087822480561854) [Y8 X9 X12 Y13]
+ (0.006087822480561854) [X8 Y9 Y12 X13]
+ (0.006509361201177252) [Y0 X1 X8 Y9]
+ (0.006509361201177252) [X0 Y1 Y8 X9]
+ (0.0068881943529705844) [Y0 X1 X6 Y7]
+ (0.0068881943529705844) [X0 Y1 Y6 X7]
+ (0.006901238249797217) [Y0 Z1 Y2 Z12]
+ (0.006901238249797217) [X0 Z1 X2 Z12]
+ (0.006901238249797217) [Y1 Z2 Y3 Z13]
+ (0.006901238249797217) [X1 Z2 X3 Z13]
+ (0.007156934919856959) [Y4 X5 X8 Y9]
+ (0.007156934919856959) [X4 Y5 Y8 X9]
+ (0.007731425250775323) [Y0 X1 X10 Y11]
+ (0.007731425250775323) [X0 Y1 Y10 X11]
+ (0.008032520918821338) [Y0 Z1 Y2 Z6]
+ (0.008032520918821338) [X0 Z1 X2 Z6]
+ (0.008032520918821338) [Y1 Z2 Y3 Z7]
+ (0.008032520918821338) [X1 Z2 X3 Z7]
+ (0.009560705729135982) [Y8 X9 X10 Y11]
+ (0.009560705729135982) [X8 Y9 Y10 X11]
+ (0.011055020596132014) [Y0 Z1 Y2 Z8]
+ (0.011055020596132014) [X0 Z1 X2 Z8]
+ (0.011055020596132014) [Y1 Z2 Y3 Z9]
+ (0.011055020596132014) [X1 Z2 X3 Z9]
+ (0.011307274008848275) [Y7 Z8 Z9 Y11]
+ (0.011307274008848275) [X7 Z8 Z9 X11]
+ (0.011982389010247972) [Y4 X5 X6 Y7]
+ (0.011982389010247972) [X4 Y5 Y6 X7]
+ (0.013873381748426127) [Y6 X7 X8 Y9]
+ (0.013873381748426127) [X6 Y7 Y8 X9]
+ (0.014583648907612613) [Y0 X1 X2 Y3]
+ (0.014583648907612613) [X0 Y1 Y2 X3]
+ (0.01557720806397642) [Y2 X3 X12 Y13]
+ (0.01557720806397642) [X2 Y3 Y12 X13]
+ (0.017366118994651417) [Y6 X7 X12 Y13]
+ (0.017366118994651417) [X6 Y7 Y12 X13]
+ (0.017680067952481466) [Y4 X5 X10 Y11]
+ (0.017680067952481466) [X4 Y5 Y10 X11]
+ (0.017825140995786474) [Y6 X7 X10 Y11]
+ (0.017825140995786474) [X6 Y7 Y10 X11]
+ (0.01902824244384731) [Y3 X4 X11 Y12]
+ (0.01902824244384731) [X3 Y4 Y11 X12]
+ (0.025384657508457396) [Y2 X3 X10 Y11]
+ (0.025384657508457396) [X2 Y3 Y10 X11]
+ (0.028685183716105903) [Y10 X11 X12 Y13]
+ (0.028685183716105903) [X10 Y11 Y12 X13]
+ (0.02981242451734581) [Y6 Z7 Z8 Y10]
+ (0.02981242451734581) [X6 Z7 Z8 X10]
+ (0.02981242451734581) [Y7 Z9 Z10 Y11]
+ (0.02981242451734581) [X7 Z9 Z10 X11]
+ (0.030104623143456882) [Y6 Z7 Z9 Y10]
+ (0.030104623143456882) [X6 Z7 Z9 X10]
+ (0.030104623143456882) [Y7 Z8 Z10 Y11]
+ (0.030104623143456882) [X7 Z8 Z10 X11]
+ (0.030787505389143995) [Y6 Z8 Z9 Y10]
+ (0.030787505389143995) [X6 Z8 Z9 X10]
+ (0.031143817988967114) [Y2 X3 X6 Y7]
+ (0.031143817988967114) [X2 Y3 Y6 X7]
+ (0.03583956795335347) [Y2 X3 X4 Y5]
+ (0.03583956795335347) [X2 Y3 Y4 X5]
+ (0.03619412355904252) [Y2 X3 X8 Y9]
+ (0.03619412355904252) [X2 Y3 Y8 X9]
+ (0.038314670294803906) [Y4 X5 X12 Y13]
+ (0.038314670294803906) [X4 Y5 Y12 X13]
+ (0.10433064780651388) [Z0 Y1 Z2 Y3]
+ (0.10433064780651388) [Z0 X1 Z2 X3]
+ (-0.12133276911042302) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042302) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042298) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042298) [X3 Z4 Z5 Z6 X7]
+ (3.2020768799164785e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768799164785e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879916479e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879916479e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918821) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918821) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918821) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918821) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329053) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329053) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329053) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329053) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273183) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273183) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273183) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273183) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021204) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021204) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964616) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964616) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964616) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964616) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173026) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173026) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173026) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173026) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613925) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613925) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613925) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613925) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613925) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613925) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613925) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613925) [X5 Z6 X7 X10 Z11 X12]
+ (-0.008764827575688803) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688803) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688803) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688803) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688803) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688803) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688803) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688803) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.007306759928832941) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832941) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832941) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832941) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826905) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826905) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826905) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826905) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017348) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017348) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017348) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017348) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825096) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825096) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825096) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825096) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155212) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155212) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776299) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776299) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639194) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639194) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441864) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441864) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400115) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400115) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400115) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400115) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598900857) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598900857) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598900857) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598900857) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255636) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255636) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524524) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524524) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630166) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630166) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369679) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369679) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730484) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730484) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730484) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730484) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125544) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125544) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956716) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956716) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956716) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956716) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591281e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591281e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591281e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591281e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864555366e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864555366e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864555366e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864555366e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215681124e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215681124e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215681124e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215681124e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675884403e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675884403e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675884403e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675884403e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738484699285e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738484699285e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738484699285e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738484699285e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433139607e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433139607e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433139607e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433139607e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.97431171341104e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.97431171341104e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122158966e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122158966e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068429501e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068429501e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218090887e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218090887e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225606823e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225606823e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451833731e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451833731e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294406705e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294406705e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971304343195e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971304343195e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971304343195e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971304343195e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500168157e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500168157e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.27748319535362e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.27748319535362e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.27748319535362e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.27748319535362e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283483017713e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283483017713e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283483017713e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283483017713e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311143459e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311143459e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711273558e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711273558e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101528624e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101528624e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491257585e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491257585e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886709635e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886709635e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825415184e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825415184e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477602036525e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477602036525e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372802302e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372802302e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742262068e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742262068e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742262068e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742262068e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420174821e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420174821e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914590757e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914590757e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914590757e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914590757e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574648374e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574648374e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574648374e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574648374e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082736384e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082736384e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082736384e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082736384e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911516508e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911516508e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624734292e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624734292e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624734292e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624734292e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624734292e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624734292e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624734292e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624734292e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751053777e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751053777e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613292131924e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613292131924e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350806998e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350806998e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265652494177e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265652494177e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265652494177e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265652494177e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128892185e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128892185e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947977639e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947977639e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947977639e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947977639e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516187035106e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516187035106e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771199162e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771199162e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771199162e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771199162e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155526462e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155526462e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155526462e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155526462e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176985588e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176985588e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176985588e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176985588e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148120051e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148120051e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148120051e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148120051e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148120051e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148120051e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148120051e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148120051e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148120051e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148120051e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148120051e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148120051e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486347455e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486347455e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599234745e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599234745e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599234745e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599234745e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599234745e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599234745e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599234745e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599234745e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595256844e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595256844e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595256844e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595256844e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134175134e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134175134e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134175134e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134175134e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155526462e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155526462e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155526462e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155526462e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516187035106e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516187035106e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128892185e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128892185e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599613951525e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599613951525e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599613951525e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599613951525e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350806998e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350806998e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613292131924e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613292131924e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751053777e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751053777e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911516508e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911516508e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420174821e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420174821e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372802302e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372802302e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651908122e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651908122e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651908122e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651908122e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477602036525e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477602036525e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825415184e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825415184e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.239336321715754e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.239336321715754e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.239336321715754e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.239336321715754e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886709635e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886709635e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491257585e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491257585e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101528624e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101528624e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711273558e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711273558e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479461549258e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479461549258e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311143459e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311143459e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500168157e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500168157e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312893664077e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312893664077e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294406705e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294406705e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559521262e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559521262e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218090887e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218090887e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068429501e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068429501e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122158966e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122158966e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.97431171341104e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.97431171341104e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611106997) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611106997) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611106997) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611106997) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916663) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916663) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499265) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499265) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499265) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499265) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125544) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125544) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213713) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213713) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213713) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213713) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440534) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440534) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440534) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440534) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369679) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369679) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630166) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630166) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524524) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524524) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339256) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339256) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339256) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339256) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496506) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496506) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496506) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496506) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441864) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441864) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639194) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639194) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776299) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776299) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155212) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155212) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221675) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221675) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221675) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221675) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109506) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109506) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109506) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109506) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921576) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921576) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921576) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921576) [X5 Z6 X7 X11 Z12 X13]
+ (0.008890731522694624) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694624) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694624) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694624) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158505) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158505) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158505) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158505) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671538) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671538) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671538) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671538) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542667) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542667) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542667) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542667) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848275) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848275) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130938) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130938) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130938) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130938) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015588250102380186) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380186) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380186) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380186) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937561) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937561) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937561) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937561) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303996) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303996) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303996) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303996) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535505) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535505) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535505) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535505) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535505) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535505) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535505) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535505) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068987) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068987) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068987) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068987) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068987) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068987) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068987) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068987) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114946) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114946) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114946) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114946) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844565) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844565) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844565) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844565) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143995) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143995) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129812) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129812) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780775) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780775) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780775) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780775) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661366) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661366) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661366) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661366) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928402144e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928402144e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.63127792840214e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.63127792840214e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860068536807e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860068536807e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860068536803e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860068536803e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274327701378271) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378271) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378272) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378272) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638318) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638318) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638318) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638318) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982184) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982184) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982184) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982184) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289345) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289345) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289345) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289345) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039318051947197646) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197646) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197646) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197646) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312456) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312456) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624863) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624863) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624863) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624863) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190551) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190551) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190551) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190551) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602684) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602684) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602684) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602684) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890995) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890995) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890995) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890995) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693034) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693034) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529033) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529033) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013112) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013112) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600954) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600954) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600954) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600954) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251617) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384731) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384731) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01602460368917948) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917948) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.014603704729162139) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162139) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173024) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173024) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.009841749246962558) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962558) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847368) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847368) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847368) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847368) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102391) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102391) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0073067599288329415) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.0073067599288329415) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561349) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561349) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017348) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017348) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109506) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109506) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400115) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400115) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832898) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832898) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832898) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832898) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423555) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423555) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423555) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423555) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025563) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025563) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806628) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806628) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806628) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806628) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524524) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524524) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524524) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524524) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.000958165583669661) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.000958165583669661) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.000958165583669661) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.000958165583669661) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.000958165583669661) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.000958165583669661) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.000958165583669661) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.000958165583669661) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569580585) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569580585) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549626) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549626) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549626) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549626) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591281e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591281e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530563106e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530563106e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530563106e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530563106e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795213956e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795213956e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795213956e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795213956e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775210477e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775210477e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775210477e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775210477e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467533814e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467533814e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467533814e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467533814e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669209012e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669209012e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669209012e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669209012e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833651708e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833651708e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833651708e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833651708e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736469809e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736469809e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736469809e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736469809e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220387406685e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220387406685e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220387406685e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220387406685e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147199889e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147199889e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147199889e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147199889e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225606823e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225606823e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451833731e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451833731e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954292878414e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954292878414e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954292878414e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954292878414e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954292878414e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954292878414e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954292878414e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954292878414e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203339244e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203339244e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203339244e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203339244e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045897445e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045897445e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045897445e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045897445e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980924287e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220980924287e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980924287e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220980924287e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365699885e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365699885e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365699885e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365699885e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769942918e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174769942918e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769942918e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174769942918e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676225237e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676225237e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676225237e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676225237e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676225237e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676225237e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676225237e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676225237e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825415184e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825415184e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825415184e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825415184e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288864428e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288864428e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288864428e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288864428e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104171076e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104171076e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104171076e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104171076e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975231915e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975231915e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207072835e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207072835e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744657892e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744657892e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447180142837e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447180142837e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447180142837e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447180142837e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896777677823e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896777677823e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108721593e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108721593e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108721593e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108721593e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350806998e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350806998e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350806998e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350806998e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565249418e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565249418e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595756972e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595756972e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595756972e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595756972e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947977639e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947977639e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155526462e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155526462e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595256843e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595256843e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.53717809603314e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.53717809603314e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.53717809603314e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.53717809603314e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595256843e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595256843e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350649731611e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649731611e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649731611e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350649731611e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555730454e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555730454e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555730454e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555730454e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155526462e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155526462e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947977639e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947977639e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565249418e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565249418e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896777677823e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896777677823e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744657892e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744657892e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207072835e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207072835e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975231915e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975231915e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886709635e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886709635e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886709635e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886709635e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435209357e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435209357e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435209357e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435209357e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951467289e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951467289e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951467289e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951467289e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400401399e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400401399e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400401399e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400401399e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400401399e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400401399e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400401399e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400401399e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019089812e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019089812e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019089812e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019089812e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019089812e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019089812e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019089812e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019089812e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500168157e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500168157e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500168157e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500168157e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893664073e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893664073e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559521262e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559521262e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591281e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591281e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569580585) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569580585) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840928) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840928) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840928) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840928) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.000594022154300525) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.000594022154300525) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.000594022154300525) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.000594022154300525) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.000594022154300525) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.000594022154300525) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.000594022154300525) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.000594022154300525) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125546) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125546) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125546) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125546) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907497) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907497) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907497) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907497) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496615) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496615) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496615) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496615) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.002261966062482356) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482356) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482356) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482356) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482356) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482356) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482356) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482356) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619324) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619324) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619324) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619324) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400115) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400115) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914304) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914304) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914304) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914304) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182559) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182559) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182559) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182559) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660391) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660391) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660391) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660391) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660391) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660391) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660391) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660391) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803878) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803878) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803878) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803878) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076824) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076824) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076824) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076824) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109506) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109506) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839374) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839374) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839374) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839374) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017348) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017348) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960916) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960916) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960916) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960916) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561349) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561349) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.0073067599288329415) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0073067599288329415) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102391) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102391) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962558) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962558) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.014564531231173024) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173024) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162139) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162139) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.01602460368917948) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917948) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384731) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384731) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251617) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251617) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129812) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129812) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615611) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615611) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615611) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615611) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023015) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023015) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767023) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036496) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036496) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036496) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036496) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986364) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986364) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986364) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986364) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635021) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635021) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635021) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635021) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214035) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214035) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214035) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214035) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312456) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312456) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0349033433736617) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0349033433736617) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0349033433736617) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0349033433736617) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382998) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382998) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693038) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693038) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529037) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529037) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013116) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013116) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314743) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314743) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314743) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314743) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589883) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589883) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589883) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589883) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917948) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917948) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917948) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917948) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.01031148248983172) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983172) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983172) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01031148248983172) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962558) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962558) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962558) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962558) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209855) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209855) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209855) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209855) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454861) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454861) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454861) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454861) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454861) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454861) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454861) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454861) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102391) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102391) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102391) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102391) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776299) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776299) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336961) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336961) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.00380406617172855) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172855) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172855) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406617172855) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0033566705638328974) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328974) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423554) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423554) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015984) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015984) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369679) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369679) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123979) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123979) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168942) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168942) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168942) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168942) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024543) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024543) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487742) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487742) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975682) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975682) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549626) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549626) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.14162522115553e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.14162522115553e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.14162522115553e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.14162522115553e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736469809e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736469809e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311143459e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311143459e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711273558e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711273558e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706408921e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706408921e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071398558e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071398558e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203339244e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203339244e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562312065e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562312065e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507321183e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507321183e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507321183e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507321183e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102938068e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102938068e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102938068e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102938068e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198877943e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198877943e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198877943e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198877943e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198877943e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198877943e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198877943e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198877943e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985785876e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985785876e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985785876e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985785876e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986211024e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986211024e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986211024e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986211024e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104171073e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104171073e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464770875e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464770875e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464770875e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464770875e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464770875e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464770875e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464770875e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464770875e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422060292e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422060292e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422060292e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422060292e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422060292e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422060292e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422060292e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422060292e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475211101597e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475211101597e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475211101597e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475211101597e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308443243e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308443243e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308443243e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308443243e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308443243e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308443243e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308443243e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308443243e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935957569714e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935957569714e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815445351174e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815445351174e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555730454e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555730454e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350649731611e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350649731611e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243605778e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243605778e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243605778e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243605778e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243605778e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243605778e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243605778e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243605778e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792999998e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792999998e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253792999998e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253792999998e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554769266e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554769266e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554769266e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554769266e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350649731611e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350649731611e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182972951e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182972951e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182972951e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182972951e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494098095e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494098095e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494098095e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494098095e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555730454e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555730454e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052260148e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052260148e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052260148e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052260148e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815445351174e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815445351174e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935957569714e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935957569714e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.09225061607668e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061607668e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.09225061607668e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061607668e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.09225061607668e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061607668e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.09225061607668e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061607668e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978541773443e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978541773443e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978541773443e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978541773443e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095173115e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095173115e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095173115e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095173115e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.24697442537013e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.24697442537013e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.24697442537013e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.24697442537013e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.24697442537013e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.24697442537013e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.24697442537013e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.24697442537013e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104171073e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104171073e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562312065e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562312065e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203339244e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203339244e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071398558e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071398558e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576032483e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576032483e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011641092e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011641092e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011641092e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011641092e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706408921e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706408921e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711273558e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711273558e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311143459e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311143459e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671239075e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671239075e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671239075e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671239075e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736469809e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736469809e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721972102e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721972102e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721972102e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721972102e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327470282e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327470282e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327470282e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327470282e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505019584856e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505019584856e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505019584856e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505019584856e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.4279886564119976e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.4279886564119976e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.4279886564119976e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.4279886564119976e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718050013e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718050013e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718050013e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718050013e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348071074e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348071074e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579337066e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579337066e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579337066e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579337066e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112187014e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112187014e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112187014e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112187014e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549626) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549626) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955039) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955039) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955039) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955039) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975682) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975682) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569580585) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580585) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569580585) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580585) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487742) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487742) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908992) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908992) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908992) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908992) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024543) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024543) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730452) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730452) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730452) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730452) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123979) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123979) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369679) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369679) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415912) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415912) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415912) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415912) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423554) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423554) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328974) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328974) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003876470899336961) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336961) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776299) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776299) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278129) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278129) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278129) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278129) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226902) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226902) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226902) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226902) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410007) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410007) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410007) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410007) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561349) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561349) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561349) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561349) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979675) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979675) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979675) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979675) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908941) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908941) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908941) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908941) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162139) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162139) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162139) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162139) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363797) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363797) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363797) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363797) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363797) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363797) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363797) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363797) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733862135) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733862135) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527260738e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527260738e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527260738e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527260738e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002705) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002705) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002706) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002706) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251617) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031148248983172) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031148248983172) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209855) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209855) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00759746402977059) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00759746402977059) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00759746402977059) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00759746402977059) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676597) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676597) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676597) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676597) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285503) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285503) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219377) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219377) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219377) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219377) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415912) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415912) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093994) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093994) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093994) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093994) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015984) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015984) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587123) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587123) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587123) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587123) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587123) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587123) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587123) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587123) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123979) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123979) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123979) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123979) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00122233780815384) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00122233780815384) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00122233780815384) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00122233780815384) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00122233780815384) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00122233780815384) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00122233780815384) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00122233780815384) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562717) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562717) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562717) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562717) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453128436e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453128436e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071398558e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071398558e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071398558e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071398558e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562312065e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562312065e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562312065e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562312065e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298077424e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298077424e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298077424e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298077424e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230049385e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230049385e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230049385e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230049385e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037103078e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037103078e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037103078e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037103078e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213132418e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213132418e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213132418e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213132418e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413731309e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413731309e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975231915e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975231915e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.87662165818915e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.87662165818915e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.87662165818915e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.87662165818915e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207072835e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207072835e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896777677823e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896777677823e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531939771e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531939771e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531939771e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531939771e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714589299264e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714589299264e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884346112e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884346112e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884346112e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884346112e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754724058e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754724058e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754724058e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754724058e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929463053e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641929463053e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315445232e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315445232e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315445232e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315445232e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641929463053e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641929463053e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815445351174e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815445351174e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815445351174e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815445351174e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589299264e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714589299264e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896777677823e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896777677823e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390474449e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390474449e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390474449e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390474449e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207072835e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207072835e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975231915e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975231915e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413731309e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413731309e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487385375e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487385375e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577104275e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577104275e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577104275e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577104275e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576032483e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576032483e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706408921e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706408921e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706408921e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706408921e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348071074e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348071074e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735241345e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735241345e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735241345e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735241345e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692951772e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692951772e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692951772e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692951772e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487742) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487742) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487742) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487742) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024543) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024543) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024543) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024543) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441885) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441885) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441885) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441885) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924534) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924534) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924534) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924534) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.00220096406950046) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00220096406950046) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00220096406950046) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00220096406950046) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980288) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980288) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980288) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980288) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980288) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980288) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980288) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980288) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415912) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415912) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285503) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285503) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369616) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369616) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369616) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369616) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464714) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464714) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464714) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464714) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209855) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209855) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031148248983172) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01031148248983172) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251617) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251617) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386214) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386214) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015964062e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015964062e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009015964058e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015964058e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0029841661681219373) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219373) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975682) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975682) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453128436e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453128436e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577104275e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577104275e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413731309e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413731309e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413731309e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413731309e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641929463053e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929463053e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929463053e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929463053e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589299264e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714589299264e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589299264e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714589299264e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487385374e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487385374e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577104275e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577104275e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975682) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975682) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219373) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219373) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
  (-73.13873231352532) [I0]
+ (-0.18066792656583402) [Z6]
+ (-0.18066792656583397) [Z7]
+ (-0.15961432501809905) [Z4]
+ (-0.15961432501809902) [Z5]
+ (0.17419956155055677) [Z2]
+ (0.17419956155055683) [Z3]
+ (0.22757269005453512) [Z0]
+ (0.22757269005453531) [Z1]
+ (-8.194261371630153e-06) [Y4 Y6]
+ (-8.194261371630153e-06) [X4 X6]
+ (7.954413175635487e-06) [Y5 Y7]
+ (7.954413175635487e-06) [X5 X7]
+ (0.11270386920332201) [Z4 Z6]
+ (0.11270386920332201) [Z5 Z7]
+ (0.11952438964682659) [Z0 Z4]
+ (0.11952438964682659) [Z1 Z5]
+ (0.13401715261963693) [Z0 Z6]
+ (0.13401715261963693) [Z1 Z7]
+ (0.13734953064261313) [Z0 Z5]
+ (0.13734953064261313) [Z1 Z4]
+ (0.13766872645852563) [Z2 Z4]
+ (0.13766872645852563) [Z3 Z5]
+ (0.14138905291942794) [Z4 Z7]
+ (0.14138905291942794) [Z5 Z6]
+ (0.14722943218766155) [Z2 Z5]
+ (0.14722943218766155) [Z3 Z4]
+ (0.14926355147388876) [Z4 Z5]
+ (0.14973486803496916) [Z2 Z6]
+ (0.14973486803496916) [Z3 Z7]
+ (0.15138327161428833) [Z0 Z7]
+ (0.15138327161428833) [Z1 Z6]
+ (0.15435748657223616) [Z6 Z7]
+ (0.15582269051553102) [Z2 Z7]
+ (0.15582269051553102) [Z3 Z6]
+ (0.16756653265461266) [Z0 Z2]
+ (0.16756653265461266) [Z1 Z3]
+ (0.19392534613270182) [Z0 Z1]
+ (-7.03788751072894e-06) [Y5 Z6 Y7]
+ (-7.03788751072894e-06) [X5 Z6 X7]
+ (-7.0378875107289386e-06) [Y4 Z5 Y6]
+ (-7.0378875107289386e-06) [X4 Z5 X6]
+ (-0.028685183716105896) [Y4 Y5 X6 X7]
+ (-0.028685183716105896) [X4 X5 Y6 Y7]
+ (-0.01782514099578655) [Y0 Y1 X4 X5]
+ (-0.01782514099578655) [X0 X1 Y4 Y5]
+ (-0.01736611899465138) [Y0 Y1 X6 X7]
+ (-0.01736611899465138) [X0 X1 Y6 Y7]
+ (-0.013873381748426063) [Y0 Y1 X2 X3]
+ (-0.013873381748426063) [X0 X1 Y2 Y3]
+ (-0.009560705729135895) [Y2 Y3 X4 X5]
+ (-0.009560705729135895) [X2 X3 Y4 Y5]
+ (-0.00608782248056185) [Y2 Y3 X6 X7]
+ (-0.00608782248056185) [X2 X3 Y6 Y7]
+ (-0.00029219862611101473) [Y1 Y2 X3 X4]
+ (-0.00029219862611101473) [X1 X2 Y3 Y4]
+ (-8.194261371630153e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261371630153e-06) [Z4 X5 Z6 X7]
+ (-2.8909678813565456e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678813565456e-06) [Z0 X5 Z6 X7]
+ (-2.8909678813565456e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678813565456e-06) [Z1 X4 Z5 X6]
+ (-1.8551201214376817e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201214376817e-06) [Z0 X4 Z5 X6]
+ (-1.8551201214376817e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201214376817e-06) [Z1 X5 Z6 X7]
+ (-1.597317197690853e-06) [Z2 Y4 Z5 Y6]
+ (-1.597317197690853e-06) [Z2 X4 Z5 X6]
+ (-1.597317197690853e-06) [Z3 Y5 Z6 Y7]
+ (-1.597317197690853e-06) [Z3 X5 Z6 X7]
+ (-1.0358477599188637e-06) [Y0 X1 X5 Y6]
+ (-1.0358477599188637e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477599188637e-06) [X0 X1 X5 X6]
+ (-1.0358477599188637e-06) [X0 Y1 Y5 X6]
+ (-9.344557775818752e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557775818752e-07) [Z2 X5 Z6 X7]
+ (-9.344557775818752e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557775818752e-07) [Z3 X4 Z5 X6]
+ (6.628614201089777e-07) [Y2 X3 X5 Y6]
+ (6.628614201089777e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201089777e-07) [X2 X3 X5 X6]
+ (6.628614201089777e-07) [X2 Y3 Y5 X6]
+ (7.954413175635487e-06) [Y4 Z5 Y6 Z7]
+ (7.954413175635487e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611101473) [Y1 X2 X3 Y4]
+ (0.00029219862611101473) [X1 Y2 Y3 X4]
+ (0.00608782248056185) [Y2 X3 X6 Y7]
+ (0.00608782248056185) [X2 Y3 Y6 X7]
+ (0.009560705729135895) [Y2 X3 X4 Y5]
+ (0.009560705729135895) [X2 Y3 Y4 X5]
+ (0.011307274008848126) [Y1 Z2 Z3 Y5]
+ (0.011307274008848126) [X1 Z2 Z3 X5]
+ (0.013873381748426063) [Y0 X1 X2 Y3]
+ (0.013873381748426063) [X0 Y1 Y2 X3]
+ (0.01736611899465138) [Y0 X1 X6 Y7]
+ (0.01736611899465138) [X0 Y1 Y6 X7]
+ (0.01782514099578655) [Y0 X1 X4 Y5]
+ (0.01782514099578655) [X0 Y1 Y4 X5]
+ (0.028685183716105896) [Y4 X5 X6 Y7]
+ (0.028685183716105896) [X4 Y5 Y6 X7]
+ (0.029812424517345847) [Y0 Z1 Z2 Y4]
+ (0.029812424517345847) [X0 Z1 Z2 X4]
+ (0.029812424517345847) [Y1 Z3 Z4 Y5]
+ (0.029812424517345847) [X1 Z3 Z4 X5]
+ (0.030104623143456865) [Y0 Z1 Z3 Y4]
+ (0.030104623143456865) [X0 Z1 Z3 X4]
+ (0.030104623143456865) [Y1 Z2 Z4 Y5]
+ (0.030104623143456865) [X1 Z2 Z4 X5]
+ (0.030787505389143974) [Y0 Z2 Z3 Y4]
+ (0.030787505389143974) [X0 Z2 Z3 X4]
+ (0.04375263801066029) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066029) [X1 Z2 Z3 Z4 X5]
+ (0.043752638010660296) [Y0 Z1 Z2 Z3 Y4]
+ (0.043752638010660296) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231172993) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172993) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172993) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172993) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848108348e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848108348e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848108348e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848108348e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659451896196e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659451896196e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971304892496e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971304892496e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971304892496e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971304892496e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131454998265132e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131454998265132e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831954859666e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831954859666e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831954859666e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831954859666e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348281835e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348281835e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348281835e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348281835e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477599188637e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477599188637e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201089777e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201089777e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350032833e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350032833e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350032833e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350032833e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201089777e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201089777e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477599188637e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477599188637e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131454998265132e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131454998265132e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.1839325589720936e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.1839325589720936e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611101473) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611101473) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611101473) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611101473) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671522) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671522) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671522) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671522) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848126) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848126) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844516) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844516) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844516) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844516) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143974) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143974) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549846799e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549846799e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396549846792e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549846792e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231172993) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172993) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659451896196e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659451896196e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350032833e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350032833e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350032833e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350032833e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131454998265124e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131454998265124e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131454998265124e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131454998265124e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932558972093e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932558972093e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172993) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172993) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
(-46.46390678868896+0j) [] +
(-0.014583648907612545+0j) [X0 X1 Y2 Y3] +
(-3.5707613288145284e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.0056526209780173075+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209792+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939577877406e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613288145284e-07+0j) [X0 X1 X3 X4] +
(-0.0056526209780173075+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209792+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577877404e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186804+0j) [X0 X1 Y4 Y5] +
(-2.4473231286679557e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.86776510451712e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285364+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231286679557e-07+0j) [X0 X1 X5 X6] +
(-7.86776510451712e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285364+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970554+0j) [X0 X1 Y6 Y7] +
(-7.735036880588565e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783553473384e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880588565e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783553473384e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0065093612011772346+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775269+0j) [X0 X1 Y10 Y11] +
(5.62785191161646e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.62785191161646e-07+0j) [X0 X1 X11 X12] +
(-0.00528377648840296+0j) [X0 X1 Y12 Y13] +
(0.014583648907612545+0j) [X0 Y1 Y2 X3] +
(3.5707613288145284e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.0056526209780173075+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209792+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939577877406e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613288145284e-07+0j) [X0 Y1 Y3 X4] +
(-0.0056526209780173075+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209792+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577877404e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186804+0j) [X0 Y1 Y4 X5] +
(2.4473231286679557e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.86776510451712e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285364+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231286679557e-07+0j) [X0 Y1 Y5 X6] +
(-7.86776510451712e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285364+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970554+0j) [X0 Y1 Y6 X7] +
(7.735036880588565e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783553473384e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880588565e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783553473384e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0065093612011772346+0j) [X0 Y1 Y8 X9] +
(0.007731425250775269+0j) [X0 Y1 Y10 X11] +
(-5.62785191161646e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.62785191161646e-07+0j) [X0 Y1 Y11 X12] +
(0.00528377648840296+0j) [X0 Y1 Y12 X13] +
(0.12507032579771496+0j) [X0 Z1 X2] +
(-1.9332412769932247e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524337+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123827+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471459129719e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412769932247e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524337+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123827+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471459129719e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315555+0j) [X0 Z1 X2 Z3] +
(-1.5510539176110114e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376508488345e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770556+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480082483e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128987050364e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676573+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631455+0j) [X0 Z1 X2 Z4] +
(-1.3807781480082485e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393086427874e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824586943+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480082485e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393086427874e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824586943+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.001105903769189613+0j) [X0 Z1 X2 Z5] +
(0.005708495985960901+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332103770232e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253795781045e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076809+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305986601487e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0080325209188213+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005244+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773244854362e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005244+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244854362e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666083+0j) [X0 Z1 X2 Z7] +
(0.011055020596131969+0j) [X0 Z1 X2 Z8] +
(0.00292976867475095+0j) [X0 Z1 X2 Z9] +
(-6.418291574918073e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914964681e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.0035552901955042022+0j) [X0 Z1 X2 Z10] +
(-1.1076325599960671e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599960671e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018411733+0j) [X0 Z1 X2 Z11] +
(0.006901238249797207+0j) [X0 Z1 X2 Z12] +
(0.0023262306231580034+0j) [X0 Z1 X2 Z13] +
(-3.568247521437982e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939856+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716556005723e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840928+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253794795946e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441842+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389678407575e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217877+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199845559e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118626+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155218+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776284+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975917174e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0051144738316603764+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692465483673e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381018+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630292+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744970412e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624920206e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639203+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441842+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389678407575e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217877+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199845559e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118626+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155218+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776284+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975917174e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0051144738316603764+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692465483673e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381018+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630292+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744970412e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624920206e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639203+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.202076879426141e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125398+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024461+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125398+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024461+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.291969486201104e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978541916337e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441863+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.684915095363469e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004403+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153685407e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616354051e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798012+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616354051e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798012+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961056277e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310132957184e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.001303800478812692+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.0039898414566192884+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197743172361e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823477+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823477+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453083484997e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217403717e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536652376263e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562535+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806597+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.839420915368541e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029757204+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538255+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289477605584e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446596873636e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369412+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.000958165583669656+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565027455e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.839420915368541e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029757204+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538255+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289477605584e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446596873636e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369412+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.000958165583669656+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565027455e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378204+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487806+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641928971152e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487806+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641928971152e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255497+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.00463697666118253+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.001280306097349654+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943053404968e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282185302497e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.0053799371558393445+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425982259e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425982259e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803862+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914288+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907466+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287494489347e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.003356670563832876+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303548238+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207452009e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422624823e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0032675138544235416+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.003356670563832876+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303548238+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207452009e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422624823e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0032675138544235416+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.0038764708993369525+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341414175751e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369525+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341414175751e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002319+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231016206+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.00422081397004644+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245088+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219316+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219316+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009013877273e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476488257546e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658413429e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213303558e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230729925+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.904599884084015e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409945+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298259767e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278058+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037350188e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226839+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079230247303e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213654+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221159098e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731754198087e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339057+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908553+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325315224395e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.606071867687948e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496481+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389547379+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309318259007e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332622702136e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440477+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.001452884321416909+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.67040239095562e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651355+0j) [X0 X2] +
(3.117447945835581e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129785+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.05859198873386174+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453554105e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612545+0j) [Y0 X1 X2 Y3] +
(3.5707613288145284e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.0056526209780173075+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209792+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939577877406e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613288145284e-07+0j) [Y0 X1 X3 Y4] +
(-0.0056526209780173075+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209792+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577877404e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186804+0j) [Y0 X1 X4 Y5] +
(2.4473231286679557e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.86776510451712e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285364+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231286679557e-07+0j) [Y0 X1 X5 Y6] +
(-7.86776510451712e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285364+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970554+0j) [Y0 X1 X6 Y7] +
(7.735036880588565e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783553473384e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880588565e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783553473384e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0065093612011772346+0j) [Y0 X1 X8 Y9] +
(0.007731425250775269+0j) [Y0 X1 X10 Y11] +
(-5.62785191161646e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.62785191161646e-07+0j) [Y0 X1 X11 Y12] +
(0.00528377648840296+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612545+0j) [Y0 Y1 X2 X3] +
(-3.5707613288145284e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.0056526209780173075+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209792+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939577877406e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613288145284e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.0056526209780173075+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209792+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577877404e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186804+0j) [Y0 Y1 X4 X5] +
(-2.4473231286679557e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.86776510451712e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285364+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231286679557e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.86776510451712e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285364+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970554+0j) [Y0 Y1 X6 X7] +
(-7.735036880588565e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783553473384e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880588565e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783553473384e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0065093612011772346+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775269+0j) [Y0 Y1 X10 X11] +
(5.62785191161646e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.62785191161646e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.00528377648840296+0j) [Y0 Y1 X12 X13] +
(-3.568247521437982e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939856+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840928+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253794795946e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716556005723e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579771496+0j) [Y0 Z1 Y2] +
(-1.9332412769932247e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524337+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123827+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471459129719e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412769932247e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524337+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123827+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471459129719e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315555+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781480082483e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128987050364e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676573+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539176110114e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376508488345e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770556+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631455+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480082485e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393086427874e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824586943+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480082485e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393086427874e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824586943+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.001105903769189613+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076809+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305986601487e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960901+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253795781045e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332103770232e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0080325209188213+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005244+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773244854362e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005244+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244854362e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666083+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596131969+0j) [Y0 Z1 Y2 Z8] +
(0.00292976867475095+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914964681e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574918073e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.0035552901955042022+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599960671e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599960671e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018411733+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797207+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231580034+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441842+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389678407575e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217877+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199845559e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118626+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155218+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776284+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975917174e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0051144738316603764+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692465483673e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381018+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630292+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744970412e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624920206e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639203+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441842+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389678407575e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217877+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199845559e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118626+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155218+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776284+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975917174e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0051144738316603764+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692465483673e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381018+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630292+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744970412e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624920206e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639203+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562535+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806597+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.202076879426141e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125398+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024461+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125398+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024461+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.291969486201104e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.684915095363469e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004403+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978541916337e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441863+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153685407e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616354051e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798012+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616354051e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798012+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961056277e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310132957184e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.0039898414566192884+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.001303800478812692+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197743172361e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823477+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823477+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453083484997e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217403717e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536652376263e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.839420915368541e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029757204+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538255+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289477605584e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446596873636e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369412+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.000958165583669656+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565027455e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.839420915368541e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029757204+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538255+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289477605584e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446596873636e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369412+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.000958165583669656+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565027455e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287494489347e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378204+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487806+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641928971152e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487806+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641928971152e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255497+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.00463697666118253+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.001280306097349654+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282185302497e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943053404968e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.0053799371558393445+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425982259e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425982259e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803862+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914288+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907466+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.003356670563832876+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303548238+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207452009e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422624823e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0032675138544235416+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.003356670563832876+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303548238+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207452009e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422624823e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0032675138544235416+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.0038764708993369525+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341414175751e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369525+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341414175751e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002319+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231016206+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.00422081397004644+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245088+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219316+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219316+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009013877273e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476488257546e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658413429e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213303558e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230729925+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.904599884084015e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409945+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298259767e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278058+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037350188e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226839+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079230247303e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213654+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221159098e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731754198087e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339057+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908553+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325315224395e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.606071867687948e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496481+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389547379+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309318259007e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332622702136e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440477+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.001452884321416909+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.67040239095562e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651355+0j) [Y0 Y2] +
(3.117447945835581e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129785+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.05859198873386174+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453554105e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111777+0j) [Z0] +
(0.10433064780651355+0j) [Z0 X1 Z2 X3] +
(3.117447945835581e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129785+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.05859198873386174+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453554105e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651355+0j) [Z0 Y1 Z2 Y3] +
(3.117447945835581e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129785+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.05859198873386174+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453554105e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860513+0j) [Z0 Z1] +
(-8.337746752646863e-07+0j) [Z0 X2 Z3 X4] +
(-0.02711503684527311+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214017+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735996538e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746752646863e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.02711503684527311+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214017+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735996538e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830387+0j) [Z0 Z2] +
(-1.1908508081461392e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329042+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950634999+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603693784274e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508081461392e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329042+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950634999+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603693784274e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.25129445674591644+0j) [Z0 Z3] +
(-3.099349243488519e-06+0j) [Z0 X4 Z5 X6] +
(-1.531680879636969e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.099349243488519e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.531680879636969e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.1966177089034214+0j) [Z0 Z4] +
(-3.3440815563553145e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585306821407e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.3440815563553145e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585306821407e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.19936354537360818+0j) [Z0 Z5] +
(0.056084681246613664+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209670213253e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613664+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209670213253e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017214+0j) [Z0 Z6] +
(0.05600733087780778+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851834678519e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780778+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851834678519e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314272+0j) [Z0 Z7] +
(0.27232518306605696+0j) [Z0 Z8] +
(0.2788345442672342+0j) [Z0 Z9] +
(-2.1776646052486255e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646052486255e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364257+0j) [Z0 Z10] +
(-1.6148794140869793e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794140869793e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441785+0j) [Z0 Z11] +
(0.21102659849791522+0j) [Z0 Z12] +
(0.21631037498631817+0j) [Z0 Z13] +
(1.9332412769932247e-07+0j) [X1 X2 Y3 Y4] +
(0.0022939566113524337+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553123827+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471459129719e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441842+0j) [X1 X2 X4 X5] +
(-8.091637199845559e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118626+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389678407575e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217877+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155217+0j) [X1 X2 X6 X7] +
(0.0051144738316603764+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465483673e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776284+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975917174e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381018+0j) [X1 X2 X8 X9] +
(-0.0017992194936630292+0j) [X1 X2 X10 X11] +
(-5.287660624920206e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744970412e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639204+0j) [X1 X2 X12 X13] +
(-1.9332412769932247e-07+0j) [X1 Y2 Y3 X4] +
(-0.0022939566113524337+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553123827+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471459129719e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441842+0j) [X1 Y2 Y4 X5] +
(-8.091637199845559e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118626+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389678407575e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217877+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155217+0j) [X1 Y2 Y6 X7] +
(0.0051144738316603764+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465483673e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776284+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975917174e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381018+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630292+0j) [X1 Y2 Y10 X11] +
(-5.287660624920206e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744970412e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639204+0j) [X1 Y2 Y12 X13] +
(0.12507032579771496+0j) [X1 Z2 X3] +
(-1.3807781480082485e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393086427874e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824586943+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480082485e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393086427874e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824586943+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001105903769189613+0j) [X1 Z2 X3 Z4] +
(-1.5510539176110114e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376508488345e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770556+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480082483e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128987050364e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676573+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631455+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005244+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773244854362e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005244+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244854362e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666083+0j) [X1 Z2 X3 Z6] +
(0.005708495985960901+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332103770232e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253795781045e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076809+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305986601487e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0080325209188213+0j) [X1 Z2 X3 Z7] +
(0.00292976867475095+0j) [X1 Z2 X3 Z8] +
(0.011055020596131969+0j) [X1 Z2 X3 Z9] +
(-1.1076325599960671e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599960671e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018411733+0j) [X1 Z2 X3 Z10] +
(-6.418291574918073e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914964681e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.0035552901955042022+0j) [X1 Z2 X3 Z11] +
(0.0023262306231580034+0j) [X1 Z2 X3 Z12] +
(0.006901238249797207+0j) [X1 Z2 X3 Z13] +
(-3.568247521437982e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939856+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716556005723e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840928+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253794795946e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125398+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024461+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209153685407e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538255+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029757204+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289477605587e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446596873638e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.000958165583669656+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369412+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565027455e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125398+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024461+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209153685407e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538255+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029757204+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289477605587e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446596873638e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.000958165583669656+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369412+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565027455e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768794261396e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616354051e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798012+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616354051e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798012+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978541916337e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441863+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.684915095363469e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004403+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153685407e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310132957184e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961056277e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823477+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823477+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453083484997e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.001303800478812692+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.0039898414566192884+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197743172361e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536652376263e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217403717e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562535+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806597+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487806+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641928971152e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832876+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303548238+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422624823e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207452009e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235416+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487806+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641928971152e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832876+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303548238+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422624823e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207452009e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235416+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378203+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.001280306097349654+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.00463697666118253+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425982259e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425982259e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803862+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943053404968e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282185302497e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.0053799371558393445+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907466+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914288+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287494489347e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336952+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341414175751e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336952+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341414175751e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.002984166168121931+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.002984166168121931+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002317+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245088+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.00422081397004644+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009013877276e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476488257546e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213303558e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.002141361223101621+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658413429e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409945+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298259767e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230729925+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.904599884084015e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226839+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079230247303e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00277902679902555+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278058+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037350188e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339057+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908553+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325315224395e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.291969486201104e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213654+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221159098e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731754198087e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332622702136e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440477+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.001452884321416909+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.67040239095562e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312315555+0j) [X1 X3] +
(3.606071867687948e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496481+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389547379+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309318259007e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412769932247e-07+0j) [Y1 X2 X3 Y4] +
(-0.0022939566113524337+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553123827+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471459129719e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441842+0j) [Y1 X2 X4 Y5] +
(-8.091637199845559e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118626+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389678407575e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217877+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155217+0j) [Y1 X2 X6 Y7] +
(0.0051144738316603764+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465483673e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776284+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975917174e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381018+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630292+0j) [Y1 X2 X10 Y11] +
(-5.287660624920206e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744970412e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639204+0j) [Y1 X2 X12 Y13] +
(1.9332412769932247e-07+0j) [Y1 Y2 X3 X4] +
(0.0022939566113524337+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553123827+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471459129719e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441842+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199845559e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118626+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389678407575e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217877+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155217+0j) [Y1 Y2 Y6 Y7] +
(0.0051144738316603764+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465483673e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776284+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975917174e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381018+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630292+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624920206e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744970412e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639204+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521437982e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939856+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840928+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253794795946e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716556005723e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579771496+0j) [Y1 Z2 Y3] +
(-1.3807781480082485e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393086427874e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824586943+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480082485e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393086427874e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824586943+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001105903769189613+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781480082483e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128987050364e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676573+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539176110114e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376508488345e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770556+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631455+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005244+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773244854362e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005244+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244854362e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666083+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076809+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305986601487e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960901+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253795781045e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332103770232e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0080325209188213+0j) [Y1 Z2 Y3 Z7] +
(0.00292976867475095+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596131969+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599960671e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599960671e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018411733+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914964681e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574918073e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.0035552901955042022+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231580034+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797207+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125398+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024461+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209153685407e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538255+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029757204+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289477605587e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446596873638e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.000958165583669656+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369412+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565027455e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125398+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024461+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209153685407e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538255+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029757204+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289477605587e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446596873638e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.000958165583669656+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369412+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565027455e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562535+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806597+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768794261396e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616354051e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798012+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616354051e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798012+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.684915095363469e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004403+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978541916337e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441863+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153685407e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310132957184e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961056277e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823477+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823477+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453083484997e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.0039898414566192884+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.001303800478812692+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197743172361e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536652376263e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217403717e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487806+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641928971152e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832876+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303548238+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422624823e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207452009e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235416+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487806+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641928971152e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832876+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303548238+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422624823e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207452009e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235416+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287494489347e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378203+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.001280306097349654+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.00463697666118253+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425982259e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425982259e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803862+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282185302497e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943053404968e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.0053799371558393445+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907466+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914288+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336952+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341414175751e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336952+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341414175751e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.002984166168121931+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.002984166168121931+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002317+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245088+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.00422081397004644+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009013877276e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476488257546e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213303558e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.002141361223101621+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658413429e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409945+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298259767e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230729925+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.904599884084015e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226839+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079230247303e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00277902679902555+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278058+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037350188e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339057+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908553+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325315224395e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.291969486201104e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213654+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221159098e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731754198087e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332622702136e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440477+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.001452884321416909+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.67040239095562e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315555+0j) [Y1 Y3] +
(3.606071867687948e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496481+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389547379+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309318259007e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111777+0j) [Z1] +
(-1.1908508081461392e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329042+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950634999+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603693784274e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508081461392e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329042+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950634999+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603693784274e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.25129445674591644+0j) [Z1 Z2] +
(-8.337746752646863e-07+0j) [Z1 X3 Z4 X5] +
(-0.02711503684527311+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214017+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735996538e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746752646863e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.02711503684527311+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214017+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735996538e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830387+0j) [Z1 Z3] +
(-3.3440815563553145e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585306821407e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.3440815563553145e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585306821407e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.19936354537360818+0j) [Z1 Z4] +
(-3.099349243488519e-06+0j) [Z1 X5 Z6 X7] +
(-1.531680879636969e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.099349243488519e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.531680879636969e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.1966177089034214+0j) [Z1 Z5] +
(0.05600733087780778+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851834678519e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780778+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851834678519e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314272+0j) [Z1 Z6] +
(0.056084681246613664+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209670213253e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613664+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209670213253e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017214+0j) [Z1 Z7] +
(0.2788345442672342+0j) [Z1 Z8] +
(0.27232518306605696+0j) [Z1 Z9] +
(-1.6148794140869793e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794140869793e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441785+0j) [Z1 Z10] +
(-2.1776646052486255e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646052486255e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364257+0j) [Z1 Z11] +
(0.21631037498631817+0j) [Z1 Z12] +
(0.21102659849791522+0j) [Z1 Z13] +
(-0.03583956795335338+0j) [X2 X3 Y4 Y5] +
(-2.1990516183194272e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.360956320406747e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831776+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183194272e-07+0j) [X2 X3 X5 X6] +
(-2.360956320406747e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831776+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967152+0j) [X2 X3 Y6 Y7] +
(0.005368659358109542+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350635226549e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109542+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350635226549e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904256+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457337+0j) [X2 X3 Y10 Y11] +
(2.172669101608488e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.172669101608488e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976444+0j) [X2 X3 Y12 Y13] +
(0.03583956795335338+0j) [X2 Y3 Y4 X5] +
(2.1990516183194272e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.360956320406747e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831776+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183194272e-07+0j) [X2 Y3 Y5 X6] +
(-2.360956320406747e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831776+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967152+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109542+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350635226549e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109542+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350635226549e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904256+0j) [X2 Y3 Y8 X9] +
(0.025384657508457337+0j) [X2 Y3 Y10 X11] +
(-2.172669101608488e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.172669101608488e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976444+0j) [X2 Y3 Y12 X13] +
(-3.887051672029392e-06+0j) [X2 Z3 X4] +
(-0.00514339176882516+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962603+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706375759e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00514339176882516+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962603+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706375759e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117381489e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489516306585e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908925+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780961165145e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411216446e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534391682602e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420193758153e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.01929956057936374+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420193758153e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.01929956057936374+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098303297e-06+0j) [X2 Z3 X4 Z7] +
(2.186842378170691e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052993808082e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380184+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.0053248352342217045+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.158656432240728e-06+0j) [X2 Z3 X4 Z10] +
(0.02435307767806895+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.02435307767806895+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707501076778e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541848578665e-06+0j) [X2 Z3 X4 Z12] +
(8.814937307332358e-06+0j) [X2 Z3 X4 Z13] +
(1.62885324368195e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796759+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158481+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.45484244899859e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463114271077e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.01925750509525157+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930677451575e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454814+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895371978774e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.64305106883605e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.01902824244384725+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.00876482757568877+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122474492e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.45484244899859e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463114271077e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.01925750509525157+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930677451575e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454814+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895371978774e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.64305106883605e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.01902824244384725+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.00876482757568877+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122474492e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042223+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023883+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815467387567e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023883+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815467387567e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0259961775980211+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.0058051889898268864+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646093+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288602568e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231089037034e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956169+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184006635695e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184006635695e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130976+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499313+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901096+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.5614471796988647e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819208+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226596+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507115539402e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429523827e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840041+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819208+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226596+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507115539402e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429523827e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840041+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162118+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.874299071559859e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162118+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.874299071559859e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702285+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946563874483e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946563874483e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354692934+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314666+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.01709155315589881+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.002446497155415856+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.002446497155415856+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.775950527538398e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676576177658e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327883746e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671496298e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.039359168022053005+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793812903e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.024755463292890887+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526722253044e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.02143381072160088+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350502136613e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624762+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.4279886568104884e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001663879878490863+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942815+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560119585352e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334298+0j) [X2 Z3 Z5 X6] +
(-0.02873077955190542+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.9358677183342945e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167406674063e-06+0j) [X2 X4] +
(0.0004956762314916312+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831236+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348489256e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335338+0j) [Y2 X3 X4 Y5] +
(2.1990516183194272e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.360956320406747e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831776+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183194272e-07+0j) [Y2 X3 X5 Y6] +
(-2.360956320406747e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831776+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967152+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109542+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350635226549e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109542+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350635226549e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904256+0j) [Y2 X3 X8 Y9] +
(0.025384657508457337+0j) [Y2 X3 X10 Y11] +
(-2.172669101608488e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.172669101608488e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976444+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335338+0j) [Y2 Y3 X4 X5] +
(-2.1990516183194272e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.360956320406747e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831776+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183194272e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.360956320406747e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831776+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967152+0j) [Y2 Y3 X6 X7] +
(0.005368659358109542+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350635226549e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109542+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350635226549e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904256+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457337+0j) [Y2 Y3 X10 X11] +
(2.172669101608488e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.172669101608488e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976444+0j) [Y2 Y3 X12 X13] +
(1.62885324368195e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796759+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158481+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051672029392e-06+0j) [Y2 Z3 Y4] +
(-0.00514339176882516+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962603+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706375759e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00514339176882516+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962603+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706375759e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117381489e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780961165145e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411216446e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489516306585e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908925+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534391682602e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420193758153e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.01929956057936374+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420193758153e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.01929956057936374+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098303297e-06+0j) [Y2 Z3 Y4 Z7] +
(2.186842378170691e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052993808082e-07+0j) [Y2 Z3 Y4 Z9] +
(0.0053248352342217045+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380184+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.158656432240728e-06+0j) [Y2 Z3 Y4 Z10] +
(0.02435307767806895+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.02435307767806895+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707501076778e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541848578665e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937307332358e-06+0j) [Y2 Z3 Y4 Z13] +
(1.45484244899859e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463114271077e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.01925750509525157+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930677451575e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454814+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895371978774e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.64305106883605e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.01902824244384725+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.00876482757568877+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122474492e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.45484244899859e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463114271077e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.01925750509525157+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930677451575e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454814+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895371978774e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.64305106883605e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.01902824244384725+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.00876482757568877+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122474492e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.5614471796988647e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042223+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023883+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815467387567e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023883+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815467387567e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0259961775980211+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.0058051889898268864+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646093+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231089037034e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288602568e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956169+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184006635695e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184006635695e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130976+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499313+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901096+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819208+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226596+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507115539402e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429523827e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840041+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819208+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226596+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507115539402e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429523827e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840041+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162118+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.874299071559859e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162118+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.874299071559859e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702285+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946563874483e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946563874483e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354692934+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314666+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.01709155315589881+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.002446497155415856+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.002446497155415856+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.775950527538398e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676576177658e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327883746e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671496298e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.039359168022053005+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793812903e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.024755463292890887+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526722253044e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.02143381072160088+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350502136613e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624762+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.4279886568104884e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001663879878490863+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942815+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560119585352e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334298+0j) [Y2 Z3 Z5 Y6] +
(-0.02873077955190542+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.9358677183342945e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167406674063e-06+0j) [Y2 Y4] +
(0.0004956762314916312+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831236+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348489256e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831668+0j) [Z2] +
(1.6021167406674063e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314916312+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831236+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348489256e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167406674063e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314916312+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831236+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348489256e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1818908579075129+0j) [Z2 Z3] +
(-9.509249751323206e-07+0j) [Z2 X4 Z5 X6] +
(-4.7288431475284065e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.02459186088382987+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751323206e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.7288431475284065e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.02459186088382987+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1249580773950318+0j) [Z2 Z4] +
(-1.1708301369642633e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467935154e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366165+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301369642633e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467935154e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366165+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1607976453483852+0j) [Z2 Z5] +
(0.01902042317303995+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156049318848e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01902042317303995+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156049318848e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683196+0j) [Z2 Z6] +
(0.024389082531149492+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.011122098579619e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149492+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.011122098579619e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1685348656157991+0j) [Z2 Z7] +
(0.1507140812100826+0j) [Z2 Z8] +
(0.18690820476912515+0j) [Z2 Z9] +
(-1.0632283425148126e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283425148126e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468384+0j) [Z2 Z10] +
(1.1094407590936753e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407590936753e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314115+0j) [Z2 Z11] +
(0.1401128986535478+0j) [Z2 Z12] +
(0.15569010671752426+0j) [Z2 Z13] +
(0.0051433917688251595+0j) [X3 X4 Y5 Y6] +
(0.009841749246962605+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706375759e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424489985899e-06+0j) [X3 X4 X6 X7] +
(-1.5224930677451575e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454814+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463114271077e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.01925750509525157+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895371978774e-07+0j) [X3 X4 X8 X9] +
(-4.64305106883605e-06+0j) [X3 X4 X10 X11] +
(-0.00876482757568877+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.01902824244384725+0j) [X3 X4 Y11 Y12] +
(5.275883122474492e-06+0j) [X3 X4 X12 X13] +
(-0.0051433917688251595+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962605+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706375759e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424489985899e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930677451575e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454814+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463114271077e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.01925750509525157+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895371978774e-07+0j) [X3 Y4 Y8 X9] +
(-4.64305106883605e-06+0j) [X3 Y4 Y10 X11] +
(-0.00876482757568877+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.01902824244384725+0j) [X3 Y4 Y11 X12] +
(5.275883122474492e-06+0j) [X3 Y4 Y12 X13] +
(-3.8870516720293925e-06+0j) [X3 Z4 X5] +
(3.2118420193758153e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.01929956057936374+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420193758153e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.01929956057936374+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098303297e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489516306585e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908925+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780961165145e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411216446e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534391682602e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052993808082e-07+0j) [X3 Z4 X5 Z8] +
(2.186842378170691e-07+0j) [X3 Z4 X5 Z9] +
(0.02435307767806895+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.02435307767806895+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707501076778e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380184+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.0053248352342217045+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.158656432240728e-06+0j) [X3 Z4 X5 Z11] +
(8.814937307332358e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541848578665e-06+0j) [X3 Z4 X5 Z13] +
(1.62885324368195e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796759+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158481+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023883+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815467387567e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.01175601341981921+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226596+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429523827e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507115539402e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840041+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023883+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815467387567e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.01175601341981921+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226596+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429523827e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507115539402e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840041+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042226+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646093+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.0058051889898268864+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184006635695e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184006635695e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130976+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288602568e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231089037034e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956169+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901096+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499313+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.5614471796988647e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162118+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.874299071559859e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162118+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.874299071559859e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946563874483e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.002446497155415856+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946563874483e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.002446497155415856+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702284+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01709155315589881+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314666+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.7759505275384006e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676576177658e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671496298e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354692937+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327883746e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.024755463292890887+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526722253044e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.039359168022053005+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793812903e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624762+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.4279886568104884e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0259961775980211+0j) [X3 Z4 Z5 X7] +
(-0.02143381072160088+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350502136613e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334298+0j) [X3 Z4 Z6 X7] +
(-0.02873077955190542+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.9358677183342945e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994117381489e-07+0j) [X3 X5] +
(0.001663879878490863+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942815+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560119585352e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0051433917688251595+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962605+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706375759e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424489985899e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930677451575e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454814+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463114271077e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.01925750509525157+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895371978774e-07+0j) [Y3 X4 X8 Y9] +
(-4.64305106883605e-06+0j) [Y3 X4 X10 Y11] +
(-0.00876482757568877+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.01902824244384725+0j) [Y3 X4 X11 Y12] +
(5.275883122474492e-06+0j) [Y3 X4 X12 Y13] +
(0.0051433917688251595+0j) [Y3 Y4 X5 X6] +
(0.009841749246962605+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706375759e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424489985899e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930677451575e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454814+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463114271077e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.01925750509525157+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895371978774e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.64305106883605e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.00876482757568877+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.01902824244384725+0j) [Y3 Y4 X11 X12] +
(5.275883122474492e-06+0j) [Y3 Y4 Y12 Y13] +
(1.62885324368195e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796759+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158481+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.8870516720293925e-06+0j) [Y3 Z4 Y5] +
(3.2118420193758153e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.01929956057936374+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420193758153e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.01929956057936374+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098303297e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780961165145e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411216446e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489516306585e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908925+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534391682602e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052993808082e-07+0j) [Y3 Z4 Y5 Z8] +
(2.186842378170691e-07+0j) [Y3 Z4 Y5 Z9] +
(0.02435307767806895+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.02435307767806895+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707501076778e-06+0j) [Y3 Z4 Y5 Z10] +
(0.0053248352342217045+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380184+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.158656432240728e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937307332358e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541848578665e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023883+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815467387567e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.01175601341981921+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226596+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429523827e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507115539402e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840041+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023883+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815467387567e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.01175601341981921+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226596+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429523827e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507115539402e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840041+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.5614471796988647e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042226+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646093+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.0058051889898268864+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184006635695e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184006635695e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130976+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231089037034e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288602568e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956169+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901096+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499313+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162118+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.874299071559859e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162118+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.874299071559859e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946563874483e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.002446497155415856+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946563874483e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.002446497155415856+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702284+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01709155315589881+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314666+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.7759505275384006e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676576177658e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671496298e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354692937+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327883746e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.024755463292890887+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526722253044e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.039359168022053005+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793812903e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624762+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.4279886568104884e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0259961775980211+0j) [Y3 Z4 Z5 Y7] +
(-0.02143381072160088+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350502136613e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334298+0j) [Y3 Z4 Z6 Y7] +
(-0.02873077955190542+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.9358677183342945e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117381489e-07+0j) [Y3 Y5] +
(0.001663879878490863+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942815+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560119585352e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831677+0j) [Z3] +
(-1.1708301369642633e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467935154e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366165+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301369642633e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467935154e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366165+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1607976453483852+0j) [Z3 Z4] +
(-9.509249751323206e-07+0j) [Z3 X5 Z6 X7] +
(-4.7288431475284065e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02459186088382987+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751323206e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.7288431475284065e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02459186088382987+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1249580773950318+0j) [Z3 Z5] +
(0.024389082531149492+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.011122098579619e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149492+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.011122098579619e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1685348656157991+0j) [Z3 Z6] +
(0.01902042317303995+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156049318848e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01902042317303995+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156049318848e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683196+0j) [Z3 Z7] +
(0.18690820476912515+0j) [Z3 Z8] +
(0.1507140812100826+0j) [Z3 Z9] +
(1.1094407590936753e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407590936753e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314115+0j) [Z3 Z10] +
(-1.0632283425148126e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283425148126e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468384+0j) [Z3 Z11] +
(0.15569010671752426+0j) [Z3 Z12] +
(0.1401128986535478+0j) [Z3 Z13] +
(-0.011982389010247953+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832984+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293593880306e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832984+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293593880306e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856932+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248153+0j) [X4 X5 Y10 Y11] +
(-3.6945132947473017e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132947473017e-06+0j) [X4 X5 X11 X12] +
(-0.038314670294803836+0j) [X4 X5 Y12 Y13] +
(0.011982389010247953+0j) [X4 Y5 Y6 X7] +
(0.007306759928832984+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293593880306e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832984+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293593880306e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856932+0j) [X4 Y5 Y8 X9] +
(0.01768006795248153+0j) [X4 Y5 Y10 X11] +
(3.6945132947473017e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132947473017e-06+0j) [X4 Y5 Y11 X12] +
(0.038314670294803836+0j) [X4 Y5 Y12 X13] +
(-1.2260484988446116e-05+0j) [X4 Z5 X6] +
(-1.228333782443042e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569576145+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782443042e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569576145+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608578726677e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449080666681e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501831324952e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921524+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730673+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978286433142e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613967+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613967+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913885165294e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155966785e-06+0j) [X4 Z5 X6 Z13] +
(0.00889073152269459+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750658272e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713808609e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840898+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535488+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218562943e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750658272e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713808609e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840898+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535488+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218562943e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731887257852e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561341+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731887257852e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561341+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928905637e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.01602460368917954+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.01602460368917954+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312897290826e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622039067102e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775878337e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736811236e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736811236e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156123+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929528974+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.00961263460684725+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026786+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817865183744e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638302+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344676457958e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982168+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433725388e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.039564416322893245+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362216168429e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719748+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.92976581532243e-07+0j) [X4 X6] +
(-4.25322422585289e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601287+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247953+0j) [Y4 X5 X6 Y7] +
(0.007306759928832984+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293593880306e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832984+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293593880306e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856932+0j) [Y4 X5 X8 Y9] +
(0.01768006795248153+0j) [Y4 X5 X10 Y11] +
(3.6945132947473017e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132947473017e-06+0j) [Y4 X5 X11 Y12] +
(0.038314670294803836+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247953+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832984+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293593880306e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832984+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293593880306e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856932+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248153+0j) [Y4 Y5 X10 X11] +
(-3.6945132947473017e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132947473017e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.038314670294803836+0j) [Y4 Y5 X12 X13] +
(0.00889073152269459+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484988446116e-05+0j) [Y4 Z5 Y6] +
(-1.228333782443042e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569576145+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782443042e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569576145+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608578726677e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449080666681e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501831324952e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730673+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921524+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978286433142e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613967+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613967+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913885165294e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155966785e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750658272e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713808609e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840898+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535488+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218562943e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750658272e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713808609e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840898+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535488+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218562943e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731887257852e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561341+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731887257852e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561341+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928905637e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.01602460368917954+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.01602460368917954+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312897290826e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622039067102e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775878337e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736811236e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736811236e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156123+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929528974+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.00961263460684725+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026786+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817865183744e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638302+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344676457958e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982168+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433725388e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.039564416322893245+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362216168429e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719748+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.92976581532243e-07+0j) [Y4 Y6] +
(-4.25322422585289e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601287+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.203440228914562+0j) [Z4] +
(-5.92976581532243e-07+0j) [Z4 X5 Z6 X7] +
(-4.2532242258528905e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012866+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.92976581532243e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.2532242258528905e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012866+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985645+0j) [Z4 Z5] +
(0.018266834869375578+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174773496429e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375578+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174773496429e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1370119167404074+0j) [Z4 Z6] +
(0.010960074940542594+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468367376735e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542594+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468367376735e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065536+0j) [Z4 Z7] +
(0.14960702684445282+0j) [Z4 Z8] +
(0.15676396176430976+0j) [Z4 Z9] +
(1.8782101249345212e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101249345212e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237594+0j) [Z4 Z10] +
(-1.8163031698127802e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031698127802e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485748+0j) [Z4 Z11] +
(0.1138357367938865+0j) [Z4 Z12] +
(0.15215040708869032+0j) [Z4 Z13] +
(1.228333782443042e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.00024636437569576145+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750658272e-07+0j) [X5 X6 X8 X9] +
(5.9743117138086075e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535488+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840898+0j) [X5 X6 Y11 Y12] +
(-4.556569218562943e-06+0j) [X5 X6 X12 X13] +
(-1.228333782443042e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.00024636437569576145+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750658272e-07+0j) [X5 Y6 Y8 X9] +
(5.9743117138086075e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535488+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840898+0j) [X5 Y6 Y11 X12] +
(-4.556569218562943e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988446113e-05+0j) [X5 Z6 X7] +
(-1.8818501831324952e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449080666681e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613967+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613967+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913885165294e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921524+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730673+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978286433142e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155966785e-06+0j) [X5 Z6 X7 Z12] +
(0.00889073152269459+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731887257852e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561341+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731887257852e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561341+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.01602460368917954+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.0714807368112366e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.01602460368917954+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.0714807368112366e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928905639e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775878337e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622039067102e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156123+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929528974+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026786+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312897290826e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.00961263460684725+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344676457958e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982168+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817865183744e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638302+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362216168429e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719748+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.854060857872668e-06+0j) [X5 X7] +
(-6.290028433725388e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.039564416322893245+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782443042e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.00024636437569576145+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750658272e-07+0j) [Y5 X6 X8 Y9] +
(5.9743117138086075e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535488+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840898+0j) [Y5 X6 X11 Y12] +
(-4.556569218562943e-06+0j) [Y5 X6 X12 Y13] +
(1.228333782443042e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.00024636437569576145+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750658272e-07+0j) [Y5 Y6 Y8 Y9] +
(5.9743117138086075e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535488+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840898+0j) [Y5 Y6 X11 X12] +
(-4.556569218562943e-06+0j) [Y5 Y6 Y12 Y13] +
(0.00889073152269459+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988446113e-05+0j) [Y5 Z6 Y7] +
(-1.8818501831324952e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449080666681e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613967+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613967+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913885165294e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730673+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921524+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978286433142e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155966785e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731887257852e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561341+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731887257852e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561341+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.01602460368917954+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.0714807368112366e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.01602460368917954+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.0714807368112366e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928905639e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775878337e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622039067102e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156123+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929528974+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026786+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312897290826e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.00961263460684725+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344676457958e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982168+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817865183744e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638302+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362216168429e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719748+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060857872668e-06+0j) [Y5 Y7] +
(-6.290028433725388e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.039564416322893245+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.203440228914562+0j) [Z5] +
(0.010960074940542594+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468367376735e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542594+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468367376735e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065536+0j) [Z5 Z6] +
(0.018266834869375578+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174773496429e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375578+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174773496429e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1370119167404074+0j) [Z5 Z7] +
(0.15676396176430976+0j) [Z5 Z8] +
(0.14960702684445282+0j) [Z5 Z9] +
(-1.8163031698127802e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031698127802e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485748+0j) [Z5 Z10] +
(1.8782101249345212e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101249345212e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237594+0j) [Z5 Z11] +
(0.15215040708869032+0j) [Z5 Z12] +
(0.1138357367938865+0j) [Z5 Z13] +
(-0.013873381748426073+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786495+0j) [X6 X7 Y10 Y11] +
(-1.035847760112012e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.035847760112012e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651365+0j) [X6 X7 Y12 Y13] +
(0.013873381748426073+0j) [X6 Y7 Y8 X9] +
(0.017825140995786495+0j) [X6 Y7 Y10 X11] +
(1.035847760112012e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.035847760112012e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651365+0j) [X6 Y7 Y12 X13] +
(0.00029219862611104915+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393505924176e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611104915+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393505924176e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918852+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500322552e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500322552e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848251+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844548+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671562+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231172987+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231172987+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860072834635e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559658893e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373849056529e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283487339775e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345823+0j) [X6 Z7 Z8 X10] +
(-3.2774831959130906e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.03010462314345687+0j) [X6 Z7 Z9 X10] +
(-3.610297130972332e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143932+0j) [X6 Z8 Z9 X10] +
(-3.769659452375209e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426073+0j) [Y6 X7 X8 Y9] +
(0.017825140995786495+0j) [Y6 X7 X10 Y11] +
(1.035847760112012e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.035847760112012e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651365+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426073+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786495+0j) [Y6 Y7 X10 X11] +
(-1.035847760112012e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.035847760112012e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651365+0j) [Y6 Y7 X12 X13] +
(0.00029219862611104915+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393505924176e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611104915+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393505924176e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918852+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500322552e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500322552e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848251+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844548+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671562+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231172987+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231172987+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860072834635e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559658893e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373849056529e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283487339775e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345823+0j) [Y6 Z7 Z8 Y10] +
(-3.2774831959130906e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.03010462314345687+0j) [Y6 Z7 Z9 Y10] +
(-3.610297130972332e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143932+0j) [Y6 Z8 Z9 Y10] +
(-3.769659452375209e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.309686298861545+0j) [Z6] +
(0.030787505389143932+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.769659452375209e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143932+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.769659452375209e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1939253461327019+0j) [Z6 Z7] +
(0.16756653265461272+0j) [Z6 Z8] +
(0.18143991440303878+0j) [Z6 Z9] +
(-1.8551201217078505e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201217078505e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682675+0j) [Z6 Z10] +
(-2.890967881819863e-06+0j) [Z6 X11 Z12 X13] +
(-2.890967881819863e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261324+0j) [Z6 Z11] +
(0.13401715261963704+0j) [Z6 Z12] +
(0.15138327161428838+0j) [Z6 Z13] +
(-0.00029219862611104915+0j) [X7 X8 Y9 Y10] +
(3.3281393505924176e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611104915+0j) [X7 Y8 Y9 X10] +
(-3.3281393505924176e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.313145500322552e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231172987+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.313145500322552e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231172987+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918855+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671562+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844548+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860072834628e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559658893e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283487339775e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848251+0j) [X7 Z8 Z9 X11] +
(-6.524373849056529e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.03010462314345687+0j) [X7 Z8 Z10 X11] +
(-3.610297130972332e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345823+0j) [X7 Z9 Z10 X11] +
(-3.2774831959130906e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611104915+0j) [Y7 X8 X9 Y10] +
(-3.3281393505924176e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611104915+0j) [Y7 Y8 X9 X10] +
(3.3281393505924176e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.313145500322552e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231172987+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.313145500322552e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231172987+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918855+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671562+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844548+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860072834628e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559658893e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283487339775e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848251+0j) [Y7 Z8 Z9 Y11] +
(-6.524373849056529e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.03010462314345687+0j) [Y7 Z8 Z10 Y11] +
(-3.610297130972332e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345823+0j) [Y7 Z9 Z10 Y11] +
(-3.2774831959130906e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615445+0j) [Z7] +
(0.18143991440303878+0j) [Z7 Z8] +
(0.16756653265461272+0j) [Z7 Z9] +
(-2.890967881819863e-06+0j) [Z7 X10 Z11 X12] +
(-2.890967881819863e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261324+0j) [Z7 Z10] +
(-1.8551201217078505e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201217078505e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682675+0j) [Z7 Z11] +
(0.15138327161428838+0j) [Z7 Z12] +
(0.13401715261963704+0j) [Z7 Z13] +
(-0.009560705729135964+0j) [X8 X9 Y10 Y11] +
(6.628614202089075e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614202089074e-07+0j) [X8 X9 X11 X12] +
(-0.00608782248056186+0j) [X8 X9 Y12 Y13] +
(0.009560705729135964+0j) [X8 Y9 Y10 X11] +
(-6.628614202089075e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614202089074e-07+0j) [X8 Y9 Y11 X12] +
(0.00608782248056186+0j) [X8 Y9 Y12 X13] +
(0.009560705729135964+0j) [Y8 X9 X10 Y11] +
(-6.628614202089075e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614202089074e-07+0j) [Y8 X9 X11 Y12] +
(0.00608782248056186+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135964+0j) [Y8 Y9 X10 X11] +
(6.628614202089075e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614202089074e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.00608782248056186+0j) [Y8 Y9 X12 X13] +
(1.369352563471818+0j) [Z8] +
(-1.5973171979380216e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171979380216e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852576+0j) [Z8 Z10] +
(-9.344557777291142e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557777291142e-07+0j) [Z8 Y11 Z12 Y13] +
(0.1472294321876617+0j) [Z8 Z11] +
(0.14973486803496924+0j) [Z8 Z12] +
(-9.344557777291142e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557777291142e-07+0j) [Z9 Y10 Z11 Y12] +
(0.1472294321876617+0j) [Z9 Z10] +
(-1.5973171979380216e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171979380216e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852576+0j) [Z9 Z11] +
(0.14973486803496924+0j) [Z9 Z13] +
(-0.028685183716105896+0j) [X10 X11 Y12 Y13] +
(0.028685183716105896+0j) [X10 Y11 Y12 X13] +
(-1.072231215793839e-05+0j) [X10 Z11 X12] +
(7.954413176857991e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372798935e-06+0j) [X10 X12] +
(0.028685183716105896+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105896+0j) [Y10 Y11 X12 X13] +
(-1.072231215793839e-05+0j) [Y10 Z11 Y12] +
(7.954413176857991e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372798935e-06+0j) [Y10 Y12] +
(-8.194261372798935e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372798935e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388903+0j) [Z10 Z11] +
(0.11270386920332215+0j) [Z10 Z12] +
(0.14138905291942805+0j) [Z10 Z13] +
(-1.072231215793838e-05+0j) [X11 Z12 X13] +
(7.954413176857991e-06+0j) [X11 X13] +
(-1.072231215793838e-05+0j) [Y11 Z12 Y13] +
(7.954413176857991e-06+0j) [Y11 Y13] +
(0.14138905291942805+0j) [Z11 Z12] +
(0.11270386920332215+0j) [Z11 Z13] +
(0.8084581961720465+0j) [Z12] +
(0.15435748657223622+0j) [Z12 Z13] +
(0.8084581961720476+0j) [Z13]
  (-46.46390678868896) [I0]
+ (0.7829661725950193) [Z11]
+ (0.7829661725950194) [Z10]
+ (0.8084581961720478) [Z12]
+ (0.8084581961720488) [Z13]
+ (1.2034402289145636) [Z4]
+ (1.2034402289145638) [Z5]
+ (1.3096862988615432) [Z7]
+ (1.3096862988615434) [Z6]
+ (1.3693525634718189) [Z8]
+ (1.3693525634718193) [Z9]
+ (1.6538942226831697) [Z2]
+ (1.65389422268317) [Z3]
+ (12.412630742111755) [Z0]
+ (12.412630742111755) [Z1]
+ (-8.194261371881756e-06) [Y10 Y12]
+ (-8.194261371881756e-06) [X10 X12]
+ (-1.8540608579681101e-06) [Y5 Y7]
+ (-1.8540608579681101e-06) [X5 X7]
+ (-7.764994118851312e-07) [Y3 Y5]
+ (-7.764994118851312e-07) [X3 X5]
+ (-5.929765816042737e-07) [Y4 Y6]
+ (-5.929765816042737e-07) [X4 X6]
+ (1.6021167405725778e-06) [Y2 Y4]
+ (1.6021167405725778e-06) [X2 X4]
+ (7.954413176038822e-06) [Y11 Y13]
+ (7.954413176038822e-06) [X11 X13]
+ (0.003276971931231617) [Y1 Y3]
+ (0.003276971931231617) [X1 X3]
+ (0.10433064780651384) [Y0 Y2]
+ (0.10433064780651384) [X0 X2]
+ (0.11270386920332215) [Z10 Z12]
+ (0.11270386920332215) [Z11 Z13]
+ (0.11383573679388659) [Z4 Z12]
+ (0.11383573679388659) [Z5 Z13]
+ (0.11952438964682684) [Z6 Z10]
+ (0.11952438964682684) [Z7 Z11]
+ (0.12489990917237612) [Z4 Z10]
+ (0.12489990917237612) [Z5 Z11]
+ (0.12495807739503213) [Z2 Z4]
+ (0.12495807739503213) [Z3 Z5]
+ (0.127995024924684) [Z2 Z10]
+ (0.127995024924684) [Z3 Z11]
+ (0.13401715261963706) [Z6 Z12]
+ (0.13401715261963706) [Z7 Z13]
+ (0.13701191674040766) [Z4 Z6]
+ (0.13701191674040766) [Z5 Z7]
+ (0.13734953064261327) [Z6 Z11]
+ (0.13734953064261327) [Z7 Z10]
+ (0.1373910476268322) [Z2 Z6]
+ (0.1373910476268322) [Z3 Z7]
+ (0.13766872645852576) [Z8 Z10]
+ (0.13766872645852576) [Z9 Z11]
+ (0.14011289865354798) [Z2 Z12]
+ (0.14011289865354798) [Z3 Z13]
+ (0.14138905291942805) [Z10 Z13]
+ (0.14138905291942805) [Z11 Z12]
+ (0.1425799771248576) [Z4 Z11]
+ (0.1425799771248576) [Z5 Z10]
+ (0.1472294321876617) [Z8 Z11]
+ (0.1472294321876617) [Z9 Z10]
+ (0.14899430575065561) [Z4 Z7]
+ (0.14899430575065561) [Z5 Z6]
+ (0.149263551473889) [Z10 Z11]
+ (0.14960702684445307) [Z4 Z8]
+ (0.14960702684445307) [Z5 Z9]
+ (0.14973486803496927) [Z8 Z12]
+ (0.14973486803496927) [Z9 Z13]
+ (0.15071408121008284) [Z2 Z8]
+ (0.15071408121008284) [Z3 Z9]
+ (0.1513832716142885) [Z6 Z13]
+ (0.1513832716142885) [Z7 Z12]
+ (0.1521504070886905) [Z4 Z13]
+ (0.1521504070886905) [Z5 Z12]
+ (0.15337968243314143) [Z2 Z11]
+ (0.15337968243314143) [Z3 Z10]
+ (0.15435748657223627) [Z12 Z13]
+ (0.15569010671752442) [Z2 Z13]
+ (0.15569010671752442) [Z3 Z12]
+ (0.1558226905155311) [Z8 Z13]
+ (0.1558226905155311) [Z9 Z12]
+ (0.15676396176431) [Z4 Z9]
+ (0.15676396176431) [Z5 Z8]
+ (0.15755314797985678) [Z4 Z5]
+ (0.16079764534838561) [Z2 Z5]
+ (0.16079764534838561) [Z3 Z4]
+ (0.1675665326546128) [Z6 Z8]
+ (0.1675665326546128) [Z7 Z9]
+ (0.16853486561579936) [Z2 Z7]
+ (0.16853486561579936) [Z3 Z6]
+ (0.18143991440303892) [Z6 Z9]
+ (0.18143991440303892) [Z7 Z8]
+ (0.1818908579075134) [Z2 Z3]
+ (0.1869082047691254) [Z2 Z9]
+ (0.1869082047691254) [Z3 Z8]
+ (0.19299723935364232) [Z0 Z10]
+ (0.19299723935364232) [Z1 Z11]
+ (0.19392534613270226) [Z6 Z7]
+ (0.19661770890342142) [Z0 Z4]
+ (0.19661770890342142) [Z1 Z5]
+ (0.19936354537360826) [Z0 Z5]
+ (0.19936354537360826) [Z1 Z4]
+ (0.20072866460441763) [Z0 Z11]
+ (0.20072866460441763) [Z1 Z10]
+ (0.211026598497915) [Z0 Z12]
+ (0.211026598497915) [Z1 Z13]
+ (0.21631037498631794) [Z0 Z13]
+ (0.21631037498631794) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830384) [Z0 Z2]
+ (0.23671080783830384) [Z1 Z3]
+ (0.24164663936017203) [Z0 Z6]
+ (0.24164663936017203) [Z1 Z7]
+ (0.2485348337131426) [Z0 Z7]
+ (0.2485348337131426) [Z1 Z6]
+ (0.2512944567459165) [Z0 Z3]
+ (0.2512944567459165) [Z1 Z2]
+ (0.2723251830660566) [Z0 Z8]
+ (0.2723251830660566) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.186176373486048) [Z0 Z1]
+ (-1.2260484988779593e-05) [Y4 Z5 Y6]
+ (-1.2260484988779593e-05) [X4 Z5 X6]
+ (-1.2260484988779588e-05) [Y5 Z6 Y7]
+ (-1.2260484988779588e-05) [X5 Z6 X7]
+ (-1.0722312157027691e-05) [Y11 Z12 Y13]
+ (-1.0722312157027691e-05) [X11 Z12 X13]
+ (-1.0722312157027684e-05) [Y10 Z11 Y12]
+ (-1.0722312157027684e-05) [X10 Z11 X12]
+ (-3.8870516738333245e-06) [Y2 Z3 Y4]
+ (-3.8870516738333245e-06) [X2 Z3 X4]
+ (-3.887051673833324e-06) [Y3 Z4 Y5]
+ (-3.887051673833324e-06) [X3 Z4 X5]
+ (0.12507032579771987) [Y1 Z2 Y3]
+ (0.12507032579771987) [X1 Z2 X3]
+ (0.1250703257977199) [Y0 Z1 Y2]
+ (0.1250703257977199) [X0 Z1 X2]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.036194123559042564) [Y2 Y3 X8 X9]
+ (-0.036194123559042564) [X2 X3 Y8 Y9]
+ (-0.03583956795335349) [Y2 Y3 X4 X5]
+ (-0.03583956795335349) [X2 X3 Y4 Y5]
+ (-0.031143817988967145) [Y2 Y3 X6 X7]
+ (-0.031143817988967145) [X2 X3 Y6 Y7]
+ (-0.028685183716105907) [Y10 Y11 X12 X13]
+ (-0.028685183716105907) [X10 X11 Y12 Y13]
+ (-0.025996177598021232) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021232) [X3 Z4 Z5 X7]
+ (-0.025384657508457423) [Y2 Y3 X10 X11]
+ (-0.025384657508457423) [X2 X3 Y10 Y11]
+ (-0.019028242443847324) [Y3 Y4 X11 X12]
+ (-0.019028242443847324) [X3 X4 Y11 Y12]
+ (-0.01782514099578644) [Y6 Y7 X10 X11]
+ (-0.01782514099578644) [X6 X7 Y10 Y11]
+ (-0.01768006795248149) [Y4 Y5 X10 X11]
+ (-0.01768006795248149) [X4 X5 Y10 Y11]
+ (-0.017366118994651392) [Y6 Y7 X12 X13]
+ (-0.017366118994651392) [X6 X7 Y12 Y13]
+ (-0.015577208063976448) [Y2 Y3 X12 X13]
+ (-0.015577208063976448) [X2 X3 Y12 Y13]
+ (-0.01458364890761264) [Y0 Y1 X2 X3]
+ (-0.01458364890761264) [X0 X1 Y2 Y3]
+ (-0.013873381748426117) [Y6 Y7 X8 X9]
+ (-0.013873381748426117) [X6 X7 Y8 Y9]
+ (-0.011982389010247953) [Y4 Y5 X6 X7]
+ (-0.011982389010247953) [X4 X5 Y6 Y7]
+ (-0.009560705729135968) [Y8 Y9 X10 X11]
+ (-0.009560705729135968) [X8 X9 Y10 Y11]
+ (-0.0077314252507753155) [Y0 Y1 X10 X11]
+ (-0.0077314252507753155) [X0 X1 Y10 Y11]
+ (-0.00715693491985695) [Y4 Y5 X8 X9]
+ (-0.00715693491985695) [X4 X5 Y8 Y9]
+ (-0.006888194352970563) [Y0 Y1 X6 X7]
+ (-0.006888194352970563) [X0 X1 Y6 Y7]
+ (-0.0065093612011772346) [Y0 Y1 X8 X9]
+ (-0.0065093612011772346) [X0 X1 Y8 Y9]
+ (-0.006087822480561855) [Y8 Y9 X12 X13]
+ (-0.006087822480561855) [X8 X9 Y12 Y13]
+ (-0.005283776488402955) [Y0 Y1 X12 X13]
+ (-0.005283776488402955) [X0 X1 Y12 Y13]
+ (-0.005143391768825104) [Y3 X4 X5 Y6]
+ (-0.005143391768825104) [X3 Y4 Y5 X6]
+ (-0.004684903388155216) [Y1 X2 X6 Y7]
+ (-0.004684903388155216) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155216) [X1 X2 X6 X7]
+ (-0.004684903388155216) [X1 Y2 Y6 X7]
+ (-0.0045750076266392005) [Y1 X2 X12 Y13]
+ (-0.0045750076266392005) [Y1 Y2 Y12 Y13]
+ (-0.0045750076266392005) [X1 X2 X12 X13]
+ (-0.0045750076266392005) [X1 Y2 Y12 X13]
+ (-0.004424855449441861) [Y1 X2 X4 Y5]
+ (-0.004424855449441861) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441861) [X1 X2 X4 X5]
+ (-0.004424855449441861) [X1 Y2 Y4 X5]
+ (-0.0034795118903343256) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343256) [X2 Z3 Z5 X6]
+ (-0.0034795118903343256) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343256) [X3 Z4 Z6 X7]
+ (-0.0027458364701868137) [Y0 Y1 X4 X5]
+ (-0.0027458364701868137) [X0 X1 Y4 Y5]
+ (-0.0017992194936630064) [Y1 X2 X10 Y11]
+ (-0.0017992194936630064) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630064) [X1 X2 X10 X11]
+ (-0.0017992194936630064) [X1 Y2 Y10 X11]
+ (-0.00029219862611107176) [Y7 Y8 X9 X10]
+ (-0.00029219862611107176) [X7 X8 Y9 Y10]
+ (-8.194261371881756e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371881756e-06) [Z10 X11 Z12 X13]
+ (-7.801707500234407e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500234407e-06) [X2 Z3 X4 Z11]
+ (-7.801707500234407e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500234407e-06) [X3 Z4 X5 Z10]
+ (-4.643051068304539e-06) [Y3 X4 X10 Y11]
+ (-4.643051068304539e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068304539e-06) [X3 X4 X10 X11]
+ (-4.643051068304539e-06) [X3 Y4 Y10 X11]
+ (-4.5888551555128415e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551555128415e-06) [X4 Z5 X6 Z13]
+ (-4.5888551555128415e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551555128415e-06) [X5 Z6 X7 Z12]
+ (-4.556569217949034e-06) [Y5 X6 X12 Y13]
+ (-4.556569217949034e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217949034e-06) [X5 X6 X12 X13]
+ (-4.556569217949034e-06) [X5 Y6 Y12 X13]
+ (-3.694513294284488e-06) [Y4 X5 X11 Y12]
+ (-3.694513294284488e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294284488e-06) [X4 X5 X11 X12]
+ (-3.694513294284488e-06) [X4 Y5 Y11 X12]
+ (-3.3440815564946375e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815564946375e-06) [Z0 X5 Z6 X7]
+ (-3.3440815564946375e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815564946375e-06) [Z1 X4 Z5 X6]
+ (-3.1586564319298676e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564319298676e-06) [X2 Z3 X4 Z10]
+ (-3.1586564319298676e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564319298676e-06) [X3 Z4 X5 Z11]
+ (-3.0993492436094293e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492436094293e-06) [Z0 X4 Z5 X6]
+ (-3.0993492436094293e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492436094293e-06) [Z1 X5 Z6 X7]
+ (-2.890967881545165e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881545165e-06) [Z6 X11 Z12 X13]
+ (-2.890967881545165e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881545165e-06) [Z7 X10 Z11 X12]
+ (-2.177664604778093e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604778093e-06) [Z0 X10 Z11 X12]
+ (-2.177664604778093e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604778093e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832206647e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832206647e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832206647e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832206647e-06) [X5 Z6 X7 Z8]
+ (-1.8551201213722082e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201213722082e-06) [Z6 X10 Z11 X12]
+ (-1.8551201213722082e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201213722082e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579681103e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579681103e-06) [X4 Z5 X6 Z7]
+ (-1.8163031695933396e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031695933396e-06) [Z4 X11 Z12 X13]
+ (-1.8163031695933396e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031695933396e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285014935e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285014935e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285014935e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285014935e-06) [X5 Z6 X7 Z11]
+ (-1.6148794136459439e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794136459439e-06) [Z0 X11 Z12 X13]
+ (-1.6148794136459439e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794136459439e-06) [Z1 X10 Z11 X12]
+ (-1.5973171976609906e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171976609906e-06) [Z8 X10 Z11 X12]
+ (-1.5973171976609906e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171976609906e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490895256e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490895256e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490895256e-06) [X3 X4 X6 X7]
+ (-1.4548424490895256e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081206555e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081206555e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081206555e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081206555e-06) [X5 Z6 X7 Z9]
+ (-1.1954890100370594e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890100370594e-06) [X2 Z3 X4 Z7]
+ (-1.1954890100370594e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890100370594e-06) [X3 Z4 X5 Z6]
+ (-1.1908508085053299e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508085053299e-06) [Z0 X3 Z4 X5]
+ (-1.1908508085053299e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508085053299e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370075623e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370075623e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370075623e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370075623e-06) [Z3 X4 Z5 X6]
+ (-1.0632283422327688e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283422327688e-06) [Z2 X10 Z11 X12]
+ (-1.0632283422327688e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283422327688e-06) [Z3 X11 Z12 X13]
+ (-1.035847760172957e-06) [Y6 X7 X11 Y12]
+ (-1.035847760172957e-06) [Y6 Y7 Y11 Y12]
+ (-1.035847760172957e-06) [X6 X7 X11 X12]
+ (-1.035847760172957e-06) [X6 Y7 Y11 X12]
+ (-9.50924975180074e-07) [Z2 Y4 Z5 Y6]
+ (-9.50924975180074e-07) [Z2 X4 Z5 X6]
+ (-9.50924975180074e-07) [Z3 Y5 Z6 Y7]
+ (-9.50924975180074e-07) [Z3 X5 Z6 X7]
+ (-9.344557775150355e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775150355e-07) [Z8 X11 Z12 X13]
+ (-9.344557775150355e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775150355e-07) [Z9 X10 Z11 X12]
+ (-8.337746755778759e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755778759e-07) [Z0 X2 Z3 X4]
+ (-8.337746755778759e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755778759e-07) [Z1 X3 Z4 X5]
+ (-7.956895372911212e-07) [Y3 X4 X8 Y9]
+ (-7.956895372911212e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372911212e-07) [X3 X4 X8 X9]
+ (-7.956895372911212e-07) [X3 Y4 Y8 X9]
+ (-7.764994118851313e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118851313e-07) [X2 Z3 X4 Z5]
+ (-5.929765816042737e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816042737e-07) [Z4 X5 Z6 X7]
+ (-5.770052995880285e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995880285e-07) [X2 Z3 X4 Z9]
+ (-5.770052995880285e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995880285e-07) [X3 Z4 X5 Z8]
+ (-5.471647744429686e-07) [Y1 Y2 X11 X12]
+ (-5.471647744429686e-07) [X1 X2 Y11 Y12]
+ (-4.838052751000094e-07) [Y5 X6 X8 Y9]
+ (-4.838052751000094e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751000094e-07) [X5 X6 X8 X9]
+ (-4.838052751000094e-07) [X5 Y6 Y8 X9]
+ (-3.570761329274537e-07) [Y0 X1 X3 Y4]
+ (-3.570761329274537e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329274537e-07) [X0 X1 X3 X4]
+ (-3.570761329274537e-07) [X0 Y1 Y3 X4]
+ (-2.4473231288520805e-07) [Y0 X1 X5 Y6]
+ (-2.4473231288520805e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231288520805e-07) [X0 X1 X5 X6]
+ (-2.4473231288520805e-07) [X0 Y1 Y5 X6]
+ (-2.1990516182748844e-07) [Y2 X3 X5 Y6]
+ (-2.1990516182748844e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516182748844e-07) [X2 X3 X5 X6]
+ (-2.1990516182748844e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771635893e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771635893e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862919803e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862919803e-07) [X1 Z2 Z3 X5]
+ (1.7379332624077152e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624077152e-07) [X0 Z1 Z3 X4]
+ (1.7379332624077152e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624077152e-07) [X1 Z2 Z4 X5]
+ (1.9332412771635893e-07) [Y1 Y2 X3 X4]
+ (1.9332412771635893e-07) [X1 X2 Y3 Y4]
+ (2.1868423770309263e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423770309263e-07) [X2 Z3 X4 Z8]
+ (2.1868423770309263e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423770309263e-07) [X3 Z4 X5 Z9]
+ (2.5935343905246626e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343905246626e-07) [X2 Z3 X4 Z6]
+ (2.5935343905246626e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343905246626e-07) [X3 Z4 X5 Z7]
+ (3.606071867985898e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867985898e-07) [X0 Z1 Z2 X4]
+ (3.606071867985898e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867985898e-07) [X1 Z3 Z4 X5]
+ (5.471647744429686e-07) [Y1 X2 X11 Y12]
+ (5.471647744429686e-07) [X1 Y2 Y11 X12]
+ (5.627851911321491e-07) [Y0 X1 X11 Y12]
+ (5.627851911321491e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911321491e-07) [X0 X1 X11 X12]
+ (5.627851911321491e-07) [X0 Y1 Y11 X12]
+ (6.628614201459549e-07) [Y8 X9 X11 Y12]
+ (6.628614201459549e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201459549e-07) [X8 X9 X11 X12]
+ (6.628614201459549e-07) [X8 Y9 Y11 X12]
+ (1.1094407592107027e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407592107027e-06) [Z2 X11 Z12 X13]
+ (1.1094407592107027e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407592107027e-06) [Z3 X10 Z11 X12]
+ (1.6021167405725778e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405725778e-06) [Z2 X3 Z4 X5]
+ (1.878210124691148e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124691148e-06) [Z4 X10 Z11 X12]
+ (1.878210124691148e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124691148e-06) [Z5 X11 Z12 X13]
+ (2.1726691014434713e-06) [Y2 X3 X11 Y12]
+ (2.1726691014434713e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691014434713e-06) [X2 X3 X11 X12]
+ (2.1726691014434713e-06) [X2 Y3 Y11 X12]
+ (3.117447946210194e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946210194e-06) [X0 Z2 Z3 X4]
+ (3.5390541843337165e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541843337165e-06) [X2 Z3 X4 Z12]
+ (3.5390541843337165e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541843337165e-06) [X3 Z4 X5 Z13]
+ (4.281913884717402e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884717402e-06) [X4 Z5 X6 Z11]
+ (4.281913884717402e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884717402e-06) [X5 Z6 X7 Z10]
+ (5.275883121925174e-06) [Y3 X4 X12 Y13]
+ (5.275883121925174e-06) [Y3 Y4 Y12 Y13]
+ (5.275883121925174e-06) [X3 X4 X12 X13]
+ (5.275883121925174e-06) [X3 Y4 Y12 X13]
+ (5.9743117132188955e-06) [Y5 X6 X10 Y11]
+ (5.9743117132188955e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117132188955e-06) [X5 X6 X10 X11]
+ (5.9743117132188955e-06) [X5 Y6 Y10 X11]
+ (7.954413176038822e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176038822e-06) [X10 Z11 X12 Z13]
+ (8.81493730625889e-06) [Y2 Z3 Y4 Z13]
+ (8.81493730625889e-06) [X2 Z3 X4 Z13]
+ (8.81493730625889e-06) [Y3 Z4 Y5 Z12]
+ (8.81493730625889e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611107176) [Y7 X8 X9 Y10]
+ (0.00029219862611107176) [X7 Y8 Y9 X10]
+ (0.0004956762314916582) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916582) [X2 Z4 Z5 X6]
+ (0.0011059037691896552) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896552) [X0 Z1 X2 Z5]
+ (0.0011059037691896552) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896552) [X1 Z2 X3 Z4]
+ (0.001663879878490779) [Y2 Z3 Z4 Y6]
+ (0.001663879878490779) [X2 Z3 Z4 X6]
+ (0.001663879878490779) [Y3 Z5 Z6 Y7]
+ (0.001663879878490779) [X3 Z5 Z6 X7]
+ (0.0017560707018412077) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412077) [X0 Z1 X2 Z11]
+ (0.0017560707018412077) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412077) [X1 Z2 X3 Z10]
+ (0.002326230623158052) [Y0 Z1 Y2 Z13]
+ (0.002326230623158052) [X0 Z1 X2 Z13]
+ (0.002326230623158052) [Y1 Z2 Y3 Z12]
+ (0.002326230623158052) [X1 Z2 X3 Z12]
+ (0.0027458364701868137) [Y0 X1 X4 Y5]
+ (0.0027458364701868137) [X0 Y1 Y4 X5]
+ (0.002929768674751024) [Y0 Z1 Y2 Z9]
+ (0.002929768674751024) [X0 Z1 X2 Z9]
+ (0.002929768674751024) [Y1 Z2 Y3 Z8]
+ (0.002929768674751024) [X1 Z2 X3 Z8]
+ (0.003276971931231617) [Y0 Z1 Y2 Z3]
+ (0.003276971931231617) [X0 Z1 X2 Z3]
+ (0.003347617530666158) [Y0 Z1 Y2 Z7]
+ (0.003347617530666158) [X0 Z1 X2 Z7]
+ (0.003347617530666158) [Y1 Z2 Y3 Z6]
+ (0.003347617530666158) [X1 Z2 X3 Z6]
+ (0.003555290195504214) [Y0 Z1 Y2 Z10]
+ (0.003555290195504214) [X0 Z1 X2 Z10]
+ (0.003555290195504214) [Y1 Z2 Y3 Z11]
+ (0.003555290195504214) [X1 Z2 X3 Z11]
+ (0.005143391768825104) [Y3 Y4 X5 X6]
+ (0.005143391768825104) [X3 X4 Y5 Y6]
+ (0.005283776488402955) [Y0 X1 X12 Y13]
+ (0.005283776488402955) [X0 Y1 Y12 X13]
+ (0.005530759218631515) [Y0 Z1 Y2 Z4]
+ (0.005530759218631515) [X0 Z1 X2 Z4]
+ (0.005530759218631515) [Y1 Z2 Y3 Z5]
+ (0.005530759218631515) [X1 Z2 X3 Z5]
+ (0.006087822480561855) [Y8 X9 X12 Y13]
+ (0.006087822480561855) [X8 Y9 Y12 X13]
+ (0.0065093612011772346) [Y0 X1 X8 Y9]
+ (0.0065093612011772346) [X0 Y1 Y8 X9]
+ (0.006888194352970563) [Y0 X1 X6 Y7]
+ (0.006888194352970563) [X0 Y1 Y6 X7]
+ (0.006901238249797251) [Y0 Z1 Y2 Z12]
+ (0.006901238249797251) [X0 Z1 X2 Z12]
+ (0.006901238249797251) [Y1 Z2 Y3 Z13]
+ (0.006901238249797251) [X1 Z2 X3 Z13]
+ (0.00715693491985695) [Y4 X5 X8 Y9]
+ (0.00715693491985695) [X4 Y5 Y8 X9]
+ (0.0077314252507753155) [Y0 X1 X10 Y11]
+ (0.0077314252507753155) [X0 Y1 Y10 X11]
+ (0.008032520918821374) [Y0 Z1 Y2 Z6]
+ (0.008032520918821374) [X0 Z1 X2 Z6]
+ (0.008032520918821374) [Y1 Z2 Y3 Z7]
+ (0.008032520918821374) [X1 Z2 X3 Z7]
+ (0.009560705729135968) [Y8 X9 X10 Y11]
+ (0.009560705729135968) [X8 Y9 Y10 X11]
+ (0.011055020596132052) [Y0 Z1 Y2 Z8]
+ (0.011055020596132052) [X0 Z1 X2 Z8]
+ (0.011055020596132052) [Y1 Z2 Y3 Z9]
+ (0.011055020596132052) [X1 Z2 X3 Z9]
+ (0.011307274008848246) [Y7 Z8 Z9 Y11]
+ (0.011307274008848246) [X7 Z8 Z9 X11]
+ (0.011982389010247953) [Y4 X5 X6 Y7]
+ (0.011982389010247953) [X4 Y5 Y6 X7]
+ (0.013873381748426117) [Y6 X7 X8 Y9]
+ (0.013873381748426117) [X6 Y7 Y8 X9]
+ (0.01458364890761264) [Y0 X1 X2 Y3]
+ (0.01458364890761264) [X0 Y1 Y2 X3]
+ (0.015577208063976448) [Y2 X3 X12 Y13]
+ (0.015577208063976448) [X2 Y3 Y12 X13]
+ (0.017366118994651392) [Y6 X7 X12 Y13]
+ (0.017366118994651392) [X6 Y7 Y12 X13]
+ (0.01768006795248149) [Y4 X5 X10 Y11]
+ (0.01768006795248149) [X4 Y5 Y10 X11]
+ (0.01782514099578644) [Y6 X7 X10 Y11]
+ (0.01782514099578644) [X6 Y7 Y10 X11]
+ (0.019028242443847324) [Y3 X4 X11 Y12]
+ (0.019028242443847324) [X3 Y4 Y11 X12]
+ (0.025384657508457423) [Y2 X3 X10 Y11]
+ (0.025384657508457423) [X2 Y3 Y10 X11]
+ (0.028685183716105907) [Y10 X11 X12 Y13]
+ (0.028685183716105907) [X10 Y11 Y12 X13]
+ (0.029812424517345767) [Y6 Z7 Z8 Y10]
+ (0.029812424517345767) [X6 Z7 Z8 X10]
+ (0.029812424517345767) [Y7 Z9 Z10 Y11]
+ (0.029812424517345767) [X7 Z9 Z10 X11]
+ (0.03010462314345684) [Y6 Z7 Z9 Y10]
+ (0.03010462314345684) [X6 Z7 Z9 X10]
+ (0.03010462314345684) [Y7 Z8 Z10 Y11]
+ (0.03010462314345684) [X7 Z8 Z10 X11]
+ (0.030787505389143953) [Y6 Z8 Z9 Y10]
+ (0.030787505389143953) [X6 Z8 Z9 X10]
+ (0.031143817988967145) [Y2 X3 X6 Y7]
+ (0.031143817988967145) [X2 Y3 Y6 X7]
+ (0.03583956795335349) [Y2 X3 X4 Y5]
+ (0.03583956795335349) [X2 Y3 Y4 X5]
+ (0.036194123559042564) [Y2 X3 X8 Y9]
+ (0.036194123559042564) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.10433064780651384) [Z0 Y1 Z2 Y3]
+ (0.10433064780651384) [Z0 X1 Z2 X3]
+ (-0.12133276911042336) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042336) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042332) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042332) [X3 Z4 Z5 Z6 X7]
+ (3.202076880081378e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880081378e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076880081379e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076880081379e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491887) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491887) [X6 Z7 Z8 Z9 X10]
+ (0.2284810656491887) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491887) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329047) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329047) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329047) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329047) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527314) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527314) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527314) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527314) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802123) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802123) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964617) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964617) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964617) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964617) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613922) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613922) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613922) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613922) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613922) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613922) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613922) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613922) [X5 Z6 X7 X10 Z11 X12]
+ (-0.008764827575688793) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688793) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688793) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688793) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688793) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688793) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688793) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688793) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.007306759928832951) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832951) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832951) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832951) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826915) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826915) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826915) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826915) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017334) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017334) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017334) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017334) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825104) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825104) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825104) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825104) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155216) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155216) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776292) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776292) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0045750076266392) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750076266392) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441861) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441861) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00415879738184004) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.00415879738184004) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00415879738184004) [X3 Z4 Z5 X6 X12 X13]
+ (-0.00415879738184004) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901317) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901317) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901317) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901317) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255822) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255822) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524632) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524632) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630064) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630064) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369772) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369772) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730411) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730411) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730411) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730411) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.000853385625412551) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.000853385625412551) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956746) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956746) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956746) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956746) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591042e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591042e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591042e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591042e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864291866e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864291866e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864291866e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864291866e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215445015e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215445015e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215445015e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215445015e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675652069e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675652069e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675652069e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675652069e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848289606e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848289606e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848289606e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848289606e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432938544e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432938544e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432938544e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432938544e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713218894e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713218894e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883121925173e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883121925173e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068304539e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068304539e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217949034e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217949034e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225527007e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225527007e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594517070837e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594517070837e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294284488e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294284488e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971303075237e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971303075237e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971303075237e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971303075237e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455000891726e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455000891726e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831952457036e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831952457036e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831952457036e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831952457036e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348200433e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348200433e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348200433e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348200433e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463110211036e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463110211036e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711131131e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711131131e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691014434713e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691014434713e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490895258e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490895258e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886397974e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886397974e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825064707e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825064707e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601729569e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601729569e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372911212e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372911212e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.73319774191674e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.73319774191674e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.73319774191674e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.73319774191674e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201459546e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201459546e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914255951e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914255951e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914255951e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914255951e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574274125e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574274125e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574274125e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574274125e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082458542e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082458542e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082458542e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082458542e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911321491e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911321491e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624458501e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624458501e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624458501e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624458501e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624458501e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624458501e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624458501e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624458501e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751000094e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751000094e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329274537e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329274537e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393506182e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393506182e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265652532373e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265652532373e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265652532373e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265652532373e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231288520805e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231288520805e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480020012e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480020012e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480020012e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480020012e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516182748844e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516182748844e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771635898e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771635898e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771635898e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771635898e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155276836e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155276836e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155276836e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155276836e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176047404e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176047404e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176047404e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176047404e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781481082347e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781481082347e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781481082347e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781481082347e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781481082347e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781481082347e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781481082347e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781481082347e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781481082347e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781481082347e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781481082347e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781481082347e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862919803e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862919803e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598209449e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598209449e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598209449e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598209449e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598209449e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598209449e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598209449e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598209449e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594581986e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594581986e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594581986e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594581986e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134891995e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134891995e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134891995e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134891995e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155276836e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155276836e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155276836e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155276836e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516182748844e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516182748844e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231288520805e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231288520805e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599614912007e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599614912007e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599614912007e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599614912007e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393506182e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393506182e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329274537e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329274537e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751000094e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751000094e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911321491e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911321491e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201459546e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201459546e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372911212e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372911212e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651582391e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651582391e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651582391e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651582391e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601729569e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601729569e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825064707e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825064707e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216835629e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216835629e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216835629e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216835629e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886397974e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886397974e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490895258e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490895258e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691014434713e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691014434713e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711131131e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711131131e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946210194e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946210194e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463110211036e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463110211036e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455000891726e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455000891726e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312892603e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312892603e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294284488e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294284488e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559332569e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559332569e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217949034e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217949034e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068304539e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068304539e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883121925173e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883121925173e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713218894e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713218894e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611107176) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611107176) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611107176) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611107176) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916582) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916582) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499083) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499083) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499083) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499083) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.000853385625412551) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.000853385625412551) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213533) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213533) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213533) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213533) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440273) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440273) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440273) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440273) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369772) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369772) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630064) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630064) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524632) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524632) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133904) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133904) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133904) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133904) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.0039615607924964906) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.0039615607924964906) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.0039615607924964906) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.0039615607924964906) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441861) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441861) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750076266392) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750076266392) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776292) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776292) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155216) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155216) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221662) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221662) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221662) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221662) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109516) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109516) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109516) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109516) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921573) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921573) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921573) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921573) [X5 Z6 X7 X11 Z12 X13]
+ (0.008890731522694614) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694614) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694614) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694614) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815853) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815853) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815853) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815853) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767154) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767154) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767154) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767154) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542634) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542634) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542634) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542634) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848246) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848246) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130936) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130936) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130936) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130936) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01558825010238019) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238019) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238019) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238019) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375585) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375585) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375585) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375585) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303997) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303997) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303997) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303997) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353549) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353549) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353549) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353549) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353549) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353549) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353549) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353549) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068984) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068984) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068984) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068984) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068984) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068984) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068984) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068984) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149485) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149485) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149485) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149485) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844537) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844537) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844537) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844537) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143953) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143953) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129785) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129785) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780759) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780759) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780759) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780759) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.0560846812466135) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.0560846812466135) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.0560846812466135) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.0560846812466135) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.63127792817369e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.63127792817369e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277928173688e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928173688e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860067792742e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860067792742e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086006779274e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006779274e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013781264) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013781264) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013781264) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013781264) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638313) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638313) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638313) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638313) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982179) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982179) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982179) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982179) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289344) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289344) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289344) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289344) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039318051947197626) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197626) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197626) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197626) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831255) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831255) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624877) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624877) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624877) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624877) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905547) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905547) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905547) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905547) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026855) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026855) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026855) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026855) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693027) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693027) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013057) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013057) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160098) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160098) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160098) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160098) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251603) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251603) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847324) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847324) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.016024603689179517) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179517) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.014603704729162136) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162136) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172996) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172996) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.009841749246962591) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962591) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847338) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847338) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847338) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847338) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023897) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023897) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832951) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832951) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613475) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613475) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017334) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017334) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109516) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109516) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00415879738184004) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00415879738184004) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832892) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832892) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832892) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832892) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235533) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235533) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235533) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235533) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255822) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255822) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806635) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806635) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806635) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806635) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524632) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524632) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524632) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524632) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696579) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696579) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696579) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696579) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696579) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696579) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696579) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696579) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756958162) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756958162) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355054) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730355054) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730355054) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730355054) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591042e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591042e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305134876e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305134876e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305134876e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305134876e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794734715e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808794734715e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794734715e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808794734715e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102774861342e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102774861342e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102774861342e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102774861342e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467322553e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467322553e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467322553e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467322553e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.65220966895824e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.65220966895824e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.65220966895824e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.65220966895824e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833406602e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833406602e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833406602e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833406602e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807362811074e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807362811074e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807362811074e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807362811074e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220385802346e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220385802346e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220385802346e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220385802346e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.72884314708725e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.72884314708725e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.72884314708725e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.72884314708725e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225527007e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225527007e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594517070837e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594517070837e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429138088e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429138088e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429138088e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429138088e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429138088e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429138088e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429138088e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429138088e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202353035e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202353035e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202353035e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202353035e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045003647e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045003647e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045003647e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045003647e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980046316e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220980046316e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980046316e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220980046316e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836516062e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836516062e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836516062e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836516062e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769492913e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174769492913e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769492913e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174769492913e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675598521e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675598521e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675598521e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675598521e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675598521e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675598521e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675598521e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675598521e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825064707e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825064707e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825064707e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825064707e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288911156e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288911156e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288911156e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288911156e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104001609e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104001609e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104001609e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104001609e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974940811e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974940811e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206819759e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206819759e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744429686e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744429686e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471800695713e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471800695713e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471800695713e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471800695713e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677524473e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677524473e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231088415873e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231088415873e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231088415873e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231088415873e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393506182e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393506182e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393506182e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393506182e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265652532373e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265652532373e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935956677074e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935956677074e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935956677074e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935956677074e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480020012e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480020012e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155276836e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155276836e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594581983e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594581983e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780950193024e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780950193024e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780950193024e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780950193024e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594581983e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594581983e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350649573326e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649573326e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649573326e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350649573326e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555163882e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555163882e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555163882e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555163882e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155276836e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155276836e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480020012e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480020012e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265652532373e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265652532373e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677524473e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677524473e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744429686e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744429686e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206819759e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206819759e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974940811e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974940811e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886397974e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886397974e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886397974e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886397974e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434612513e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434612513e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434612513e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434612513e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489513947924e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489513947924e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489513947924e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489513947924e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184002469724e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184002469724e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184002469724e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184002469724e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184002469724e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184002469724e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184002469724e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184002469724e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420189546443e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189546443e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189546443e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189546443e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189546443e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189546443e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420189546443e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189546443e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455000891726e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455000891726e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455000891726e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455000891726e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312892603e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312892603e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559332569e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559332569e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591042e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591042e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756958162) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756958162) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840992) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840992) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840992) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840992) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005341) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005341) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005341) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005341) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005341) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005341) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005341) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005341) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.000853385625412551) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.000853385625412551) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.000853385625412551) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.000853385625412551) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.001043524653490731) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.001043524653490731) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.001043524653490731) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.001043524653490731) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496337) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496337) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496337) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496337) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.002261966062482353) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482353) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482353) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482353) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482353) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482353) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482353) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482353) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619331) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619331) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619331) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619331) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.00415879738184004) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00415879738184004) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914284) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914284) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914284) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914284) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.0046369766611825255) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.0046369766611825255) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.0046369766611825255) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.0046369766611825255) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.00511447383166039) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.00511447383166039) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.00511447383166039) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.00511447383166039) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.00511447383166039) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511447383166039) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511447383166039) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00511447383166039) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803848) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803848) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803848) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803848) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076826) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076826) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076826) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076826) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109516) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109516) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839352) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839352) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839352) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839352) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017334) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017334) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960925) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960925) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960925) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960925) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613475) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613475) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832951) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832951) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023897) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023897) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962591) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962591) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.014564531231172996) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172996) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162136) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162136) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.016024603689179517) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179517) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847324) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847324) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251603) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251603) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129785) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129785) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156267) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156267) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.36937089366156256) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156256) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.2816425776702297) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702297) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767022965) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022965) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0906514420703648) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703648) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703648) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703648) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863625) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863625) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863625) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863625) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635015) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635015) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635015) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635015) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214028) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214028) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214028) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214028) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831255) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831255) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366178) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366178) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366178) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366178) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830016) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830016) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830016) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830016) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693024) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693024) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952906) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952906) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013057) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013057) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314767) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314767) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314767) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314767) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589888) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589888) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589888) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589888) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179517) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179517) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179517) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179517) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.01031148248983176) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983176) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983176) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01031148248983176) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962591) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962591) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962591) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962591) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209865) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209865) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209865) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209865) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454847) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454847) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454847) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454847) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454847) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454847) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454847) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454847) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023897) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023897) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023897) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023897) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776292) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776292) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369538) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369538) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728541) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728541) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728541) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728541) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003356670563832891) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832891) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235533) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235533) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015794) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015794) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369772) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369772) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124098) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124098) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416882) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416882) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416882) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416882) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024502) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024502) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00051927434994877) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.00051927434994877) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029757426) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029757426) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355054) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730355054) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.14162522115404e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.14162522115404e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.14162522115404e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.14162522115404e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807362811074e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807362811074e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463110211036e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463110211036e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711131131e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711131131e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063746395e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063746395e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071273573e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071273573e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202353035e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202353035e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562272756e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562272756e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376506866343e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376506866343e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376506866343e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376506866343e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102592422e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102592422e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102592422e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102592422e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198533362e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198533362e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198533362e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198533362e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198533362e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198533362e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198533362e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198533362e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985478707e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985478707e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985478707e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985478707e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985857454e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985857454e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985857454e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985857454e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104001609e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104001609e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464430316e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464430316e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464430316e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464430316e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464430316e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464430316e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464430316e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464430316e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421854389e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421854389e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421854389e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421854389e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421854389e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421854389e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421854389e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421854389e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475210088895e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475210088895e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475210088895e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475210088895e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930833298e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.37673930833298e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930833298e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.37673930833298e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930833298e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.37673930833298e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.37673930833298e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.37673930833298e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935956677074e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935956677074e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815443399785e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815443399785e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555163882e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555163882e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350649573326e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350649573326e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243500024e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243500024e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243500024e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243500024e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243500024e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243500024e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243500024e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243500024e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379145272e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379145272e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379145272e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379145272e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554292162e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554292162e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554292162e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554292162e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350649573326e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350649573326e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182627623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182627623e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182627623e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182627623e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493692393e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493692393e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493692393e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493692393e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555163882e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555163882e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051473815e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051473815e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051473815e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051473815e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815443399785e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815443399785e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935956677074e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935956677074e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506159231277e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506159231277e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506159231277e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506159231277e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506159231277e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506159231277e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506159231277e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506159231277e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978540674444e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978540674444e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978540674444e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978540674444e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095063533e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095063533e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095063533e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095063533e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425082521e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425082521e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425082521e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425082521e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425082521e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425082521e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425082521e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425082521e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104001609e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104001609e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562272756e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562272756e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202353035e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202353035e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071273573e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071273573e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676575978431e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676575978431e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011528313e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011528313e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011528313e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011528313e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063746395e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063746395e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711131131e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711131131e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463110211036e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463110211036e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.84620167112384e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.84620167112384e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.84620167112384e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.84620167112384e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807362811074e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807362811074e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721834217e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721834217e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721834217e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721834217e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327351115e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327351115e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327351115e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327351115e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501803488e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501803488e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501803488e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501803488e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656237486e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656237486e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656237486e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656237486e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717902951e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717902951e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717902951e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717902951e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347867744e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347867744e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579310779e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579310779e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579310779e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579310779e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217487e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411217487e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411217487e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217487e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730355054) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730355054) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955272) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955272) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955272) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955272) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029757426) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029757426) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569581626) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569581626) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569581626) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569581626) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00051927434994877) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.00051927434994877) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909097) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909097) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909097) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909097) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024502) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024502) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532483523073068) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532483523073068) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532483523073068) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532483523073068) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124098) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124098) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369772) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369772) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158887) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158887) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158887) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158887) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235533) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235533) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832891) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832891) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0038764708993369538) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369538) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776292) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776292) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278144) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278144) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278144) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278144) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226914) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226914) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226914) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226914) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410021) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410021) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410021) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410021) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796756) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796756) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796756) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796756) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908929) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908929) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908929) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908929) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162134) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162134) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162134) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162134) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363776) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363776) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363776) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363776) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363776) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363776) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363776) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363776) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386208) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386208) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527025044e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527025044e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527025045e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527025045e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002807) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002807) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100281) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100281) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251603) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251603) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031148248983176) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031148248983176) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209865) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209865) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297706095) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297706095) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297706095) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297706095) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311877) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311877) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311877) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311877) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311877) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311877) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311877) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311877) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676615) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676615) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676615) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676615) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728541) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728541) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219347) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219347) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219347) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219347) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158895) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158895) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093993) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093993) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093993) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093993) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015794) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015794) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587312) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587312) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587312) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587312) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587312) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587312) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587312) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587312) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124098) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124098) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124098) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124098) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538377) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538377) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538377) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538377) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538377) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538377) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538377) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538377) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562632) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562632) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562632) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562632) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146306145265875e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.146306145265875e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071273573e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071273573e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071273573e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071273573e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562272756e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562272756e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562272756e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562272756e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129759229e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129759229e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129759229e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129759229e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229560156e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229560156e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229560156e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229560156e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036698918e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036698918e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036698918e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036698918e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212693421e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212693421e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212693421e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212693421e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413404829e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413404829e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974940811e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974940811e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657842564e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657842564e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657842564e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657842564e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206819759e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206819759e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677524473e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677524473e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.07673253182103e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.07673253182103e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.07673253182103e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.07673253182103e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714587805325e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714587805325e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998841874596e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998841874596e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998841874596e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998841874596e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754675013e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754675013e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754675013e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754675013e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192861236e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192861236e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316607743e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309316607743e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316607743e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309316607743e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192861236e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192861236e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815443399785e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815443399785e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815443399785e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815443399785e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587805325e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714587805325e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677524473e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677524473e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023904413063e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023904413063e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023904413063e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023904413063e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206819759e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206819759e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974940811e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974940811e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413404829e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413404829e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487045697e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487045697e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576440053e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576440053e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576440053e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576440053e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676575978431e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676575978431e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063746395e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063746395e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063746395e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063746395e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347867744e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347867744e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734769089e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734769089e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734769089e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734769089e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692413097e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692413097e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692413097e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692413097e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00051927434994877) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.00051927434994877) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.00051927434994877) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.00051927434994877) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024502) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024502) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024502) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024502) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441872) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441872) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441872) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441872) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245565) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245565) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245565) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245565) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004502) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004502) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004502) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004502) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798025) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798025) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798025) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798025) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798025) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798025) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798025) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798025) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158895) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158895) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728541) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728541) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369533) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369533) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369533) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369533) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046492) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046492) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046492) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046492) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209865) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209865) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031148248983176) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01031148248983176) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251603) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251603) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386208) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386208) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901550298e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901550298e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3987009015502976e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015502976e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0029841661681219347) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219347) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029757426) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029757426) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452658751e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452658751e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576440053e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576440053e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413404829e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413404829e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413404829e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413404829e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192861236e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192861236e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192861236e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192861236e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458780532e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458780532e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458780532e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458780532e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487045698e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487045698e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576440053e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576440053e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757426) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757426) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219347) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219347) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
  (-73.13873231352534) [I0]
+ (-0.18066792656583475) [Z7]
+ (-0.1806679265658347) [Z6]
+ (-0.15961432501810033) [Z4]
+ (-0.15961432501810024) [Z5]
+ (0.17419956155055605) [Z3]
+ (0.17419956155055613) [Z2]
+ (0.22757269005453423) [Z0]
+ (0.22757269005453448) [Z1]
+ (-8.194261372699508e-06) [Y4 Y6]
+ (-8.194261372699508e-06) [X4 X6]
+ (7.95441317675553e-06) [Y5 Y7]
+ (7.95441317675553e-06) [X5 X7]
+ (0.11270386920332215) [Z4 Z6]
+ (0.11270386920332215) [Z5 Z7]
+ (0.1195243896468268) [Z0 Z4]
+ (0.1195243896468268) [Z1 Z5]
+ (0.1340171526196369) [Z0 Z6]
+ (0.1340171526196369) [Z1 Z7]
+ (0.13734953064261327) [Z0 Z5]
+ (0.13734953064261327) [Z1 Z4]
+ (0.13766872645852582) [Z2 Z4]
+ (0.13766872645852582) [Z3 Z5]
+ (0.14138905291942805) [Z4 Z7]
+ (0.14138905291942805) [Z5 Z6]
+ (0.14722943218766182) [Z2 Z5]
+ (0.14722943218766182) [Z3 Z4]
+ (0.14926355147388906) [Z4 Z5]
+ (0.1497348680349691) [Z2 Z6]
+ (0.1497348680349691) [Z3 Z7]
+ (0.15138327161428827) [Z0 Z7]
+ (0.15138327161428827) [Z1 Z6]
+ (0.1543574865722361) [Z6 Z7]
+ (0.15582269051553094) [Z2 Z7]
+ (0.15582269051553094) [Z3 Z6]
+ (0.16756653265461263) [Z0 Z2]
+ (0.16756653265461263) [Z1 Z3]
+ (0.19392534613270193) [Z0 Z1]
+ (-7.037887510358193e-06) [Y4 Z5 Y6]
+ (-7.037887510358193e-06) [X4 Z5 X6]
+ (-7.037887510358193e-06) [Y5 Z6 Y7]
+ (-7.037887510358193e-06) [X5 Z6 X7]
+ (-0.028685183716105907) [Y4 Y5 X6 X7]
+ (-0.028685183716105907) [X4 X5 Y6 Y7]
+ (-0.01782514099578648) [Y0 Y1 X4 X5]
+ (-0.01782514099578648) [X0 X1 Y4 Y5]
+ (-0.017366118994651406) [Y0 Y1 X6 X7]
+ (-0.017366118994651406) [X0 X1 Y6 Y7]
+ (-0.0138733817484261) [Y0 Y1 X2 X3]
+ (-0.0138733817484261) [X0 X1 Y2 Y3]
+ (-0.009560705729135964) [Y2 Y3 X4 X5]
+ (-0.009560705729135964) [X2 X3 Y4 Y5]
+ (-0.00608782248056184) [Y2 Y3 X6 X7]
+ (-0.00608782248056184) [X2 X3 Y6 Y7]
+ (-0.0002921986261110622) [Y1 Y2 X3 X4]
+ (-0.0002921986261110622) [X1 X2 Y3 Y4]
+ (-8.194261372699508e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372699508e-06) [Z4 X5 Z6 X7]
+ (-2.890967881801241e-06) [Z0 Y5 Z6 Y7]
+ (-2.890967881801241e-06) [Z0 X5 Z6 X7]
+ (-2.890967881801241e-06) [Z1 Y4 Z5 Y6]
+ (-2.890967881801241e-06) [Z1 X4 Z5 X6]
+ (-1.855120121710876e-06) [Z0 Y4 Z5 Y6]
+ (-1.855120121710876e-06) [Z0 X4 Z5 X6]
+ (-1.855120121710876e-06) [Z1 Y5 Z6 Y7]
+ (-1.855120121710876e-06) [Z1 X5 Z6 X7]
+ (-1.5973171979558641e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171979558641e-06) [Z2 X4 Z5 X6]
+ (-1.5973171979558641e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171979558641e-06) [Z3 X5 Z6 X7]
+ (-1.0358477600903648e-06) [Y0 X1 X5 Y6]
+ (-1.0358477600903648e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477600903648e-06) [X0 X1 X5 X6]
+ (-1.0358477600903648e-06) [X0 Y1 Y5 X6]
+ (-9.344557777416403e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557777416403e-07) [Z2 X5 Z6 X7]
+ (-9.344557777416403e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557777416403e-07) [Z3 X4 Z5 X6]
+ (6.62861420214224e-07) [Y2 X3 X5 Y6]
+ (6.62861420214224e-07) [Y2 Y3 Y5 Y6]
+ (6.62861420214224e-07) [X2 X3 X5 X6]
+ (6.62861420214224e-07) [X2 Y3 Y5 X6]
+ (7.95441317675553e-06) [Y4 Z5 Y6 Z7]
+ (7.95441317675553e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110622) [Y1 X2 X3 Y4]
+ (0.0002921986261110622) [X1 Y2 Y3 X4]
+ (0.00608782248056184) [Y2 X3 X6 Y7]
+ (0.00608782248056184) [X2 Y3 Y6 X7]
+ (0.009560705729135964) [Y2 X3 X4 Y5]
+ (0.009560705729135964) [X2 Y3 Y4 X5]
+ (0.011307274008848208) [Y1 Z2 Z3 Y5]
+ (0.011307274008848208) [X1 Z2 Z3 X5]
+ (0.0138733817484261) [Y0 X1 X2 Y3]
+ (0.0138733817484261) [X0 Y1 Y2 X3]
+ (0.017366118994651406) [Y0 X1 X6 Y7]
+ (0.017366118994651406) [X0 Y1 Y6 X7]
+ (0.01782514099578648) [Y0 X1 X4 Y5]
+ (0.01782514099578648) [X0 Y1 Y4 X5]
+ (0.028685183716105907) [Y4 X5 X6 Y7]
+ (0.028685183716105907) [X4 Y5 Y6 X7]
+ (0.029812424517345767) [Y0 Z1 Z2 Y4]
+ (0.029812424517345767) [X0 Z1 Z2 X4]
+ (0.029812424517345767) [Y1 Z3 Z4 Y5]
+ (0.029812424517345767) [X1 Z3 Z4 X5]
+ (0.030104623143456827) [Y0 Z1 Z3 Y4]
+ (0.030104623143456827) [X0 Z1 Z3 X4]
+ (0.030104623143456827) [Y1 Z2 Z4 Y5]
+ (0.030104623143456827) [X1 Z2 Z4 X5]
+ (0.030787505389143925) [Y0 Z2 Z3 Y4]
+ (0.030787505389143925) [X0 Z2 Z3 X4]
+ (0.04375263801066015) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066015) [X0 Z1 Z2 Z3 X4]
+ (0.04375263801066015) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066015) [X1 Z2 Z3 Z4 X5]
+ (-0.014564531231173043) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231173043) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231173043) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231173043) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373849034216e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373849034216e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373849034216e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373849034216e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594523865896e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594523865896e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971309910065e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971309910065e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971309910065e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971309910065e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145500278496e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145500278496e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831959269027e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831959269027e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831959269027e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831959269027e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348755719e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348755719e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348755719e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348755719e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477600903648e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477600903648e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.62861420214224e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.62861420214224e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393506410357e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393506410357e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393506410357e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393506410357e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.62861420214224e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.62861420214224e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477600903648e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477600903648e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145500278496e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145500278496e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559570831e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559570831e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611106216) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611106216) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611106216) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611106216) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671482) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671482) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671482) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671482) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.01130727400884821) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.01130727400884821) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844527) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844527) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844527) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844527) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143925) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143925) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.1053965500020005e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.1053965500020005e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396550002e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396550002e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231173045) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231173045) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594523865896e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594523865896e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393506410357e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393506410357e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393506410357e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393506410357e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455002784964e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455002784964e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455002784964e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455002784964e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559570831e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559570831e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231173045) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231173045) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 36. tutorial_local_cost_functions.html <a name="demo35"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_local_cost_functions.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     0
Plateau'd:     1
--- New run! ---
Cost after step    20:  0.9993000
Cost after step    40:  0.9994000
Cost after step    60:  0.9995000
Cost after step    80:  0.9994000
Cost after step   100:  0.9996000
Cost after step   120:  0.9995000
Cost after step   140:  0.9986000
Cost after step   160:  0.9993000
Cost after step   180:  0.9991000
Cost after step   200:  0.9996000
Cost after step   220:  0.9985000
Cost after step   240:  0.9989000
Cost after step   260:  0.9993000
Cost after step   280:  0.9991000
Cost after step   300:  0.9990000
Cost after step   320:  0.9987000
Cost after step   340:  0.9987000
Cost after step   360:  0.9994000
Cost after step   380:  0.9985000
Cost after step   400:  0.9991000
Trained:     0
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9976000
Cost after step    40:  0.9967000
Cost after step    60:  0.9970000
Cost after step    80:  0.9964000
Cost after step   100:  0.9956000
Cost after step   120:  0.9965000
Cost after step   140:  0.9958000
Cost after step   160:  0.9931000
Cost after step   180:  0.9940000
Cost after step   200:  0.9908000
Cost after step   220:  0.9887000
Cost after step   240:  0.9834000
Cost after step   260:  0.9821000
Cost after step   280:  0.9742000
Cost after step   300:  0.9576000
Cost after step   320:  0.9330000
Trained:     1
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9998000
Cost after step    40:  0.9997000
Cost after step    60:  0.9996000
Cost after step    80:  0.9994000
Cost after step   100:  0.9994000
Cost after step   120:  0.9996000
Cost after step   140:  0.9997000
Cost after step   160:  0.9992000
Cost after step   180:  0.9991000
Cost after step   200:  0.9992000
Cost after step   220:  0.9990000
Cost after step   240:  0.9986000
Cost after step   260:  0.9989000
Cost after step   280:  0.9988000
Cost after step   300:  0.9972000
Cost after step   320:  0.9982000
Cost after step   340:  0.9981000
Cost after step   360:  0.9979000
Cost after step   380:  0.9977000
Cost after step   400:  0.9972000
Trained:     1
Plateau'd:     3
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  0.9999000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  0.9999000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  0.9998000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     1
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9950000
Cost after step    40:  0.9957000
Cost after step    60:  0.9931000
Cost after step    80:  0.9920000
Cost after step   100:  0.9925000
Cost after step   120:  0.9908000
Cost after step   140:  0.9865000
Cost after step   160:  0.9861000
Cost after step   180:  0.9846000
Cost after step   200:  0.9767000
Cost after step   220:  0.9696000
Cost after step   240:  0.9560000
Cost after step   260:  0.9276000
Trained:     2
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9989000
Cost after step    40:  0.9979000
Cost after step    60:  0.9979000
Cost after step    80:  0.9982000
Cost after step   100:  0.9984000
Cost after step   120:  0.9986000
Cost after step   140:  0.9978000
Cost after step   160:  0.9976000
Cost after step   180:  0.9967000
Cost after step   200:  0.9972000
Cost after step   220:  0.9958000
Cost after step   240:  0.9966000
Cost after step   260:  0.9966000
Cost after step   280:  0.9952000
Cost after step   300:  0.9958000
Cost after step   320:  0.9972000
Cost after step   340:  0.9953000
Cost after step   360:  0.9934000
Cost after step   380:  0.9929000
Cost after step   400:  0.9916000
Trained:     2
Plateau'd:     5
--- New run! ---
Cost after step    20:  0.9988000
Cost after step    40:  0.9980000
Cost after step    60:  0.9978000
Cost after step    80:  0.9986000
Cost after step   100:  0.9978000
Cost after step   120:  0.9977000
Cost after step   140:  0.9974000
Cost after step   160:  0.9976000
Cost after step   180:  0.9975000
Cost after step   200:  0.9970000
Cost after step   220:  0.9972000
Cost after step   240:  0.9961000
Cost after step   260:  0.9960000
Cost after step   280:  0.9953000
Cost after step   300:  0.9947000
Cost after step   320:  0.9945000
Cost after step   340:  0.9912000
Cost after step   360:  0.9916000
Cost after step   380:  0.9866000
Cost after step   400:  0.9822000
Trained:     2
Plateau'd:     6
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     7
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  0.9999000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     8
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6137000. Locality: 2
---Switching Locality---
Cost after step    20:  0.5159000. Locality: 3
---Switching Locality---
Cost after step    30:  0.7040000. Locality: 4
Cost after step    40:  0.5823000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5055000. Locality: 5
---Switching Locality---
Cost after step    60:  0.6965000. Locality: 6
Cost after step    70:  0.5823000. Locality: 6
---Switching Locality---
Cost after step    80:  0.8792000. Locality: 7
Cost after step    90:  0.7172000. Locality: 7
---Switching Locality---
Cost after step   100:  0.9741000. Locality: 8
Cost after step   110:  0.9329000. Locality: 8
Cost after step   120:  0.8278000. Locality: 8
Cost after step   130:  0.5973000. Locality: 8
Cost after step   140:  0.2649000. Locality: 8
Trained:     1
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6273000. Locality: 2
---Switching Locality---
Cost after step    20:  0.9218000. Locality: 4
Cost after step    30:  0.8247000. Locality: 4
Cost after step    40:  0.5992000. Locality: 4
---Switching Locality---
Cost after step    50:  0.6852000. Locality: 5
Cost after step    60:  0.6003000. Locality: 5
Cost after step    70:  0.5220000. Locality: 5
---Switching Locality---
Cost after step    80:  0.5100000. Locality: 6
---Switching Locality---
Cost after step    90:  0.4677000. Locality: 8
Cost after step   100:  0.1562000. Locality: 8
Trained:     2
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.5405000. Locality: 5
---Switching Locality---
Cost after step    20:  0.6024000. Locality: 6
---Switching Locality---
Cost after step    30:  0.6060000. Locality: 7
---Switching Locality---
Cost after step    40:  0.5009000. Locality: 8
Cost after step    50:  0.1676000. Locality: 8
Trained:     3
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.4871000. Locality: 1
---Switching Locality---
Cost after step    20:  0.4948000. Locality: 2
---Switching Locality---
Cost after step    30:  0.6627000. Locality: 3
---Switching Locality---
Cost after step    40:  0.5826000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5234000. Locality: 5
---Switching Locality---
Cost after step    60:  0.6394000. Locality: 7
---Switching Locality---
Cost after step    70:  0.3126000. Locality: 8
Trained:     4
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5485000. Locality: 1
---Switching Locality---
Cost after step    20:  0.6393000. Locality: 2
---Switching Locality---
Cost after step    30:  0.8061000. Locality: 3
Cost after step    40:  0.7073000. Locality: 3
Cost after step    50:  0.6144000. Locality: 3
Cost after step    60:  0.4888000. Locality: 3
---Switching Locality---
Cost after step    70:  0.6102000. Locality: 4
---Switching Locality---
Cost after step    80:  0.4909000. Locality: 5
---Switching Locality---
Cost after step    90:  0.7897000. Locality: 6
Cost after step   100:  0.4939000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7063000. Locality: 7
Cost after step   120:  0.5323000. Locality: 7
---Switching Locality---
Cost after step   130:  0.2921000. Locality: 8
Trained:     5
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.7449000. Locality: 2
Cost after step    20:  0.5005000. Locality: 2
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
---Switching Locality---
Cost after step    50:  0.5099000. Locality: 7
---Switching Locality---
Cost after step    60:  0.6587000. Locality: 8
Cost after step    70:  0.4912000. Locality: 8
Cost after step    80:  0.2440000. Locality: 8
Trained:     6
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.8510000. Locality: 3
Cost after step    20:  0.7255000. Locality: 3
Cost after step    30:  0.5811000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6300000. Locality: 4
---Switching Locality---
Cost after step    50:  0.8801000. Locality: 5
Cost after step    60:  0.7395000. Locality: 5
---Switching Locality---
Cost after step    70:  0.8948000. Locality: 6
Cost after step    80:  0.7399000. Locality: 6
---Switching Locality---
Cost after step    90:  0.7139000. Locality: 7
Cost after step   100:  0.5959000. Locality: 7
---Switching Locality---
Cost after step   110:  0.5939000. Locality: 8
Cost after step   120:  0.2906000. Locality: 8
Trained:     7
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6986000. Locality: 2
Cost after step    20:  0.6273000. Locality: 2
Cost after step    30:  0.5645000. Locality: 2
Cost after step    40:  0.5016000. Locality: 2
---Switching Locality---
Cost after step    50:  0.9048000. Locality: 3
Cost after step    60:  0.7903000. Locality: 3
Cost after step    70:  0.5948000. Locality: 3
---Switching Locality---
Cost after step    80:  0.5600000. Locality: 5
---Switching Locality---
Cost after step    90:  0.8307000. Locality: 6
Cost after step   100:  0.6442000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7075000. Locality: 7
Cost after step   120:  0.5936000. Locality: 7
---Switching Locality---
Cost after step   130:  0.7649000. Locality: 8
Cost after step   140:  0.6810000. Locality: 8
Cost after step   150:  0.5936000. Locality: 8
Cost after step   160:  0.4526000. Locality: 8
Cost after step   170:  0.1974000. Locality: 8
Trained:     8
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5488000. Locality: 1
---Switching Locality---
Cost after step    20:  0.8358000. Locality: 2
Cost after step    30:  0.7614000. Locality: 2
Cost after step    40:  0.6718000. Locality: 2
Cost after step    50:  0.4921000. Locality: 2
---Switching Locality---
Cost after step    60:  0.6433000. Locality: 4
---Switching Locality---
Cost after step    70:  0.7787000. Locality: 5
Cost after step    80:  0.5250000. Locality: 5
---Switching Locality---
Cost after step    90:  0.5270000. Locality: 6
---Switching Locality---
Cost after step   100:  0.9690000. Locality: 8
Cost after step   110:  0.9106000. Locality: 8
Cost after step   120:  0.7561000. Locality: 8
Cost after step   130:  0.4537000. Locality: 8
Cost after step   140:  0.1197000. Locality: 8
Trained:     9
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6759000. Locality: 2
---Switching Locality---
Cost after step    20:  0.8020000. Locality: 3
Cost after step    30:  0.6412000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6920000. Locality: 4
---Switching Locality---
Cost after step    50:  0.7176000. Locality: 5
Cost after step    60:  0.5973000. Locality: 5
---Switching Locality---
Cost after step    70:  0.5469000. Locality: 6
---Switching Locality---
Cost after step    80:  0.7019000. Locality: 7
Cost after step    90:  0.6006000. Locality: 7
---Switching Locality---
Cost after step   100:  0.8612000. Locality: 8
Cost after step   110:  0.7023000. Locality: 8
Cost after step   120:  0.4108000. Locality: 8
Cost after step   130:  0.1081000. Locality: 8
Trained:    10
Plateau'd:     0
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_local_cost_functions.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     0
Plateau'd:     1
--- New run! ---
Cost after step    20:  0.9993000
Cost after step    40:  0.9994000
Cost after step    60:  0.9995000
Cost after step    80:  0.9994000
Cost after step   100:  0.9996000
Cost after step   120:  0.9995000
Cost after step   140:  0.9986000
Cost after step   160:  0.9993000
Cost after step   180:  0.9991000
Cost after step   200:  0.9996000
Cost after step   220:  0.9985000
Cost after step   240:  0.9989000
Cost after step   260:  0.9993000
Cost after step   280:  0.9991000
Cost after step   300:  0.9990000
Cost after step   320:  0.9987000
Cost after step   340:  0.9987000
Cost after step   360:  0.9994000
Cost after step   380:  0.9985000
Cost after step   400:  0.9991000
Trained:     0
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9976000
Cost after step    40:  0.9967000
Cost after step    60:  0.9970000
Cost after step    80:  0.9964000
Cost after step   100:  0.9956000
Cost after step   120:  0.9965000
Cost after step   140:  0.9958000
Cost after step   160:  0.9931000
Cost after step   180:  0.9940000
Cost after step   200:  0.9908000
Cost after step   220:  0.9887000
Cost after step   240:  0.9834000
Cost after step   260:  0.9821000
Cost after step   280:  0.9742000
Cost after step   300:  0.9576000
Cost after step   320:  0.9330000
Trained:     1
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9998000
Cost after step    40:  0.9997000
Cost after step    60:  0.9996000
Cost after step    80:  0.9994000
Cost after step   100:  0.9994000
Cost after step   120:  0.9996000
Cost after step   140:  0.9997000
Cost after step   160:  0.9992000
Cost after step   180:  0.9991000
Cost after step   200:  0.9992000
Cost after step   220:  0.9990000
Cost after step   240:  0.9986000
Cost after step   260:  0.9989000
Cost after step   280:  0.9988000
Cost after step   300:  0.9972000
Cost after step   320:  0.9982000
Cost after step   340:  0.9981000
Cost after step   360:  0.9979000
Cost after step   380:  0.9977000
Cost after step   400:  0.9972000
Trained:     1
Plateau'd:     3
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  0.9999000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  0.9999000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  0.9998000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     1
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9950000
Cost after step    40:  0.9957000
Cost after step    60:  0.9931000
Cost after step    80:  0.9920000
Cost after step   100:  0.9925000
Cost after step   120:  0.9908000
Cost after step   140:  0.9865000
Cost after step   160:  0.9861000
Cost after step   180:  0.9846000
Cost after step   200:  0.9767000
Cost after step   220:  0.9696000
Cost after step   240:  0.9560000
Cost after step   260:  0.9276000
Trained:     2
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9989000
Cost after step    40:  0.9979000
Cost after step    60:  0.9979000
Cost after step    80:  0.9982000
Cost after step   100:  0.9984000
Cost after step   120:  0.9986000
Cost after step   140:  0.9978000
Cost after step   160:  0.9976000
Cost after step   180:  0.9967000
Cost after step   200:  0.9972000
Cost after step   220:  0.9958000
Cost after step   240:  0.9966000
Cost after step   260:  0.9966000
Cost after step   280:  0.9952000
Cost after step   300:  0.9958000
Cost after step   320:  0.9972000
Cost after step   340:  0.9953000
Cost after step   360:  0.9934000
Cost after step   380:  0.9929000
Cost after step   400:  0.9916000
Trained:     2
Plateau'd:     5
--- New run! ---
Cost after step    20:  0.9988000
Cost after step    40:  0.9980000
Cost after step    60:  0.9978000
Cost after step    80:  0.9986000
Cost after step   100:  0.9978000
Cost after step   120:  0.9977000
Cost after step   140:  0.9974000
Cost after step   160:  0.9976000
Cost after step   180:  0.9975000
Cost after step   200:  0.9970000
Cost after step   220:  0.9972000
Cost after step   240:  0.9961000
Cost after step   260:  0.9960000
Cost after step   280:  0.9953000
Cost after step   300:  0.9947000
Cost after step   320:  0.9945000
Cost after step   340:  0.9912000
Cost after step   360:  0.9916000
Cost after step   380:  0.9866000
Cost after step   400:  0.9822000
Trained:     2
Plateau'd:     6
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     7
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  0.9999000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     8
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6137000. Locality: 2
---Switching Locality---
Cost after step    20:  0.5159000. Locality: 3
---Switching Locality---
Cost after step    30:  0.7040000. Locality: 4
Cost after step    40:  0.5823000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5055000. Locality: 5
---Switching Locality---
Cost after step    60:  0.6965000. Locality: 6
Cost after step    70:  0.5823000. Locality: 6
---Switching Locality---
Cost after step    80:  0.8792000. Locality: 7
Cost after step    90:  0.7172000. Locality: 7
---Switching Locality---
Cost after step   100:  0.9741000. Locality: 8
Cost after step   110:  0.9329000. Locality: 8
Cost after step   120:  0.8278000. Locality: 8
Cost after step   130:  0.5973000. Locality: 8
Cost after step   140:  0.2649000. Locality: 8
Trained:     1
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6273000. Locality: 2
---Switching Locality---
Cost after step    20:  0.9218000. Locality: 4
Cost after step    30:  0.8247000. Locality: 4
Cost after step    40:  0.5992000. Locality: 4
---Switching Locality---
Cost after step    50:  0.6852000. Locality: 5
Cost after step    60:  0.6003000. Locality: 5
Cost after step    70:  0.5220000. Locality: 5
---Switching Locality---
Cost after step    80:  0.5100000. Locality: 6
---Switching Locality---
Cost after step    90:  0.4677000. Locality: 8
Cost after step   100:  0.1562000. Locality: 8
Trained:     2
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.5405000. Locality: 5
---Switching Locality---
Cost after step    20:  0.6024000. Locality: 6
---Switching Locality---
Cost after step    30:  0.6060000. Locality: 7
---Switching Locality---
Cost after step    40:  0.5009000. Locality: 8
Cost after step    50:  0.1676000. Locality: 8
Trained:     3
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.4871000. Locality: 1
---Switching Locality---
Cost after step    20:  0.4948000. Locality: 2
---Switching Locality---
Cost after step    30:  0.6627000. Locality: 3
---Switching Locality---
Cost after step    40:  0.5826000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5234000. Locality: 5
---Switching Locality---
Cost after step    60:  0.6394000. Locality: 7
---Switching Locality---
Cost after step    70:  0.3126000. Locality: 8
Trained:     4
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5485000. Locality: 1
---Switching Locality---
Cost after step    20:  0.6393000. Locality: 2
---Switching Locality---
Cost after step    30:  0.8061000. Locality: 3
Cost after step    40:  0.7073000. Locality: 3
Cost after step    50:  0.6144000. Locality: 3
Cost after step    60:  0.4888000. Locality: 3
---Switching Locality---
Cost after step    70:  0.6102000. Locality: 4
---Switching Locality---
Cost after step    80:  0.4909000. Locality: 5
---Switching Locality---
Cost after step    90:  0.7897000. Locality: 6
Cost after step   100:  0.4939000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7063000. Locality: 7
Cost after step   120:  0.5323000. Locality: 7
---Switching Locality---
Cost after step   130:  0.2921000. Locality: 8
Trained:     5
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.7449000. Locality: 2
Cost after step    20:  0.5005000. Locality: 2
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
---Switching Locality---
Cost after step    50:  0.5099000. Locality: 7
---Switching Locality---
Cost after step    60:  0.6587000. Locality: 8
Cost after step    70:  0.4912000. Locality: 8
Cost after step    80:  0.2440000. Locality: 8
Trained:     6
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.8510000. Locality: 3
Cost after step    20:  0.7255000. Locality: 3
Cost after step    30:  0.5811000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6300000. Locality: 4
---Switching Locality---
Cost after step    50:  0.8801000. Locality: 5
Cost after step    60:  0.7395000. Locality: 5
---Switching Locality---
Cost after step    70:  0.8948000. Locality: 6
Cost after step    80:  0.7399000. Locality: 6
---Switching Locality---
Cost after step    90:  0.7139000. Locality: 7
Cost after step   100:  0.5959000. Locality: 7
---Switching Locality---
Cost after step   110:  0.5939000. Locality: 8
Cost after step   120:  0.2906000. Locality: 8
Trained:     7
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6986000. Locality: 2
Cost after step    20:  0.6273000. Locality: 2
Cost after step    30:  0.5645000. Locality: 2
Cost after step    40:  0.5016000. Locality: 2
---Switching Locality---
Cost after step    50:  0.9048000. Locality: 3
Cost after step    60:  0.7903000. Locality: 3
Cost after step    70:  0.5948000. Locality: 3
---Switching Locality---
Cost after step    80:  0.5600000. Locality: 5
---Switching Locality---
Cost after step    90:  0.8307000. Locality: 6
Cost after step   100:  0.6442000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7075000. Locality: 7
Cost after step   120:  0.5936000. Locality: 7
---Switching Locality---
Cost after step   130:  0.7649000. Locality: 8
Cost after step   140:  0.6810000. Locality: 8
Cost after step   150:  0.5936000. Locality: 8
Cost after step   160:  0.4526000. Locality: 8
Cost after step   170:  0.1974000. Locality: 8
Trained:     8
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5488000. Locality: 1
---Switching Locality---
Cost after step    20:  0.8358000. Locality: 2
Cost after step    30:  0.7614000. Locality: 2
Cost after step    40:  0.6718000. Locality: 2
Cost after step    50:  0.4921000. Locality: 2
---Switching Locality---
Cost after step    60:  0.6433000. Locality: 4
---Switching Locality---
Cost after step    70:  0.7787000. Locality: 5
Cost after step    80:  0.5250000. Locality: 5
---Switching Locality---
Cost after step    90:  0.5270000. Locality: 6
---Switching Locality---
Cost after step   100:  0.9690000. Locality: 8
Cost after step   110:  0.9106000. Locality: 8
Cost after step   120:  0.7561000. Locality: 8
Cost after step   130:  0.4537000. Locality: 8
Cost after step   140:  0.1197000. Locality: 8
Trained:     9
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6759000. Locality: 2
---Switching Locality---
Cost after step    20:  0.8020000. Locality: 3
Cost after step    30:  0.6412000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6920000. Locality: 4
---Switching Locality---
Cost after step    50:  0.7176000. Locality: 5
Cost after step    60:  0.5973000. Locality: 5
---Switching Locality---
Cost after step    70:  0.5469000. Locality: 6
---Switching Locality---
Cost after step    80:  0.7019000. Locality: 7
Cost after step    90:  0.6006000. Locality: 7
---Switching Locality---
Cost after step   100:  0.8612000. Locality: 8
Cost after step   110:  0.7023000. Locality: 8
Cost after step   120:  0.4108000. Locality: 8
Cost after step   130:  0.1081000. Locality: 8
Trained:    10
 </code>
 </pre>
 </details>

---

