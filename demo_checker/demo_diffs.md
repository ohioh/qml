Last update: 2021-11-01  11:55:11 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_ensemble_multi_qpu.html](#demo0)
2. [tutorial_falqon.html](#demo1)
3. [tutorial_chemical_reactions.html](#demo2)
4. [tutorial_quantum_transfer_learning.html](#demo3)
5. [tutorial_QGAN.html](#demo4)
6. [tutorial_kernels_module.html](#demo5)
7. [tutorial_vqe_parallel.html](#demo6)
8. [tutorial_adaptive_circuits.html](#demo7)
9. [tutorial_qaoa_intro.html](#demo8)
10. [tutorial_gaussian_transformation.html](#demo9)
11. [tutorial_gbs.html](#demo10)
12. [tutorial_pasqal.html](#demo11)
13. [tutorial_variational_classifier.html](#demo12)
14. [tutorial_backprop.html](#demo13)
15. [tutorial_doubly_stochastic.html](#demo14)
16. [tutorial_mol_geo_opt.html](#demo15)
17. [tutorial_noisy_circuits.html](#demo16)
18. [tutorial_data_reuploading_classifier.html](#demo17)
19. [tutorial_vqe_spin_sectors.html](#demo18)
20. [tutorial_expressivity_fourier_series.html](#demo19)
21. [tutorial_multiclass_classification.html](#demo20)
22. [tutorial_vqe_qng.html](#demo21)
23. [tutorial_general_parshift.html](#demo22)
24. [tutorial_qnn_module_tf.html](#demo23)
25. [tutorial_vqt.html](#demo24)
26. [tutorial_vqe.html](#demo25)
27. [tutorial_local_cost_functions.html](#demo26)
28. [tutorial_quantum_metrology.html](#demo27)
29. [tutorial_rosalin.html](#demo28)
30. [tutorial_quantum_chemistry.html](#demo29)
31. [tutorial_jax_transformations.html](#demo30)
32. [tutorial_measurement_optimize.html](#demo31)
33. [tutorial_qgrnn.html](#demo32)
34. [tutorial_quanvolution.html](#demo33)
35. [tutorial_rotoselect.html](#demo34)
36. [tutorial_quantum_natural_gradient.html](#demo35)


Number of demos different/all demos: 36/54

## 1. tutorial_ensemble_multi_qpu.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.832
Training accuracy (QPU1):  0.288
Choices: [0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 1 0 1 0 0 0
Choices counts: Counter({0: 111, 1: 39})
Counter({2: 56, 0: 55})
Counter({1: 36, 0: 3})
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.824
Training accuracy (QPU1):  0.296
Choices: [0 0 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 1 0 1 0 0 0
Choices counts: Counter({0: 110, 1: 40})
Counter({0: 55, 2: 55})
Counter({1: 37, 0: 3})
```

---

## 2. tutorial_falqon.html <a name="demo1"></a>

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
/home/runner/work/qml/qml/demonstrations/tutorial_falqon.py:234: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/operation.py:730: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_falqon.py:234: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/operation.py:730: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_falqon.py:234: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/operation.py:730: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/home/runner/work/qml/qml/demonstrations/tutorial_falqon.py:418: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/operation.py:730: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
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
 </code>
 </pre>
 </details>

---

## 3. tutorial_chemical_reactions.html <a name="demo2"></a>

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

## 4. tutorial_quantum_transfer_learning.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 34%|###4      | 15.2M/44.7M [00:00<00:00, 159MB/s]
 70%|######9   | 31.2M/44.7M [00:00<00:00, 164MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 183MB/s]
Training started:
Phase: train Epoch: 1/1 Iter: 1/62 Batch time: 0.3045
Phase: train Epoch: 1/1 Iter: 2/62 Batch time: 0.2828
Phase: train Epoch: 1/1 Iter: 3/62 Batch time: 0.2791
Phase: train Epoch: 1/1 Iter: 4/62 Batch time: 0.2883
Phase: train Epoch: 1/1 Iter: 5/62 Batch time: 0.2767
Phase: train Epoch: 1/1 Iter: 6/62 Batch time: 0.2758
Phase: train Epoch: 1/1 Iter: 7/62 Batch time: 0.2744
Phase: train Epoch: 1/1 Iter: 8/62 Batch time: 0.2760
Phase: train Epoch: 1/1 Iter: 9/62 Batch time: 0.2686
Phase: train Epoch: 1/1 Iter: 10/62 Batch time: 0.2766
Phase: train Epoch: 1/1 Iter: 11/62 Batch time: 0.2768
Phase: train Epoch: 1/1 Iter: 12/62 Batch time: 0.2755
Phase: train Epoch: 1/1 Iter: 13/62 Batch time: 0.2755
Phase: train Epoch: 1/1 Iter: 14/62 Batch time: 0.2755
Phase: train Epoch: 1/1 Iter: 15/62 Batch time: 0.2767
Phase: train Epoch: 1/1 Iter: 16/62 Batch time: 0.2705
Phase: train Epoch: 1/1 Iter: 17/62 Batch time: 0.2709
Phase: train Epoch: 1/1 Iter: 18/62 Batch time: 0.2742
Phase: train Epoch: 1/1 Iter: 19/62 Batch time: 0.2806
Phase: train Epoch: 1/1 Iter: 20/62 Batch time: 0.2701
Phase: train Epoch: 1/1 Iter: 21/62 Batch time: 0.2694
Phase: train Epoch: 1/1 Iter: 22/62 Batch time: 0.2749
Phase: train Epoch: 1/1 Iter: 23/62 Batch time: 0.2763
Phase: train Epoch: 1/1 Iter: 24/62 Batch time: 0.2782
Phase: train Epoch: 1/1 Iter: 25/62 Batch time: 0.2729
Phase: train Epoch: 1/1 Iter: 26/62 Batch time: 0.2744
Phase: train Epoch: 1/1 Iter: 27/62 Batch time: 0.2768
Phase: train Epoch: 1/1 Iter: 28/62 Batch time: 0.2684
Phase: train Epoch: 1/1 Iter: 29/62 Batch time: 0.2724
Phase: train Epoch: 1/1 Iter: 30/62 Batch time: 0.2731
Phase: train Epoch: 1/1 Iter: 31/62 Batch time: 0.2779
Phase: train Epoch: 1/1 Iter: 32/62 Batch time: 0.2786
Phase: train Epoch: 1/1 Iter: 33/62 Batch time: 0.2753
Phase: train Epoch: 1/1 Iter: 34/62 Batch time: 0.2728
Phase: train Epoch: 1/1 Iter: 35/62 Batch time: 0.2756
Phase: train Epoch: 1/1 Iter: 36/62 Batch time: 0.2649
Phase: train Epoch: 1/1 Iter: 37/62 Batch time: 0.2701
Phase: train Epoch: 1/1 Iter: 38/62 Batch time: 0.2647
Phase: train Epoch: 1/1 Iter: 39/62 Batch time: 0.2722
Phase: train Epoch: 1/1 Iter: 40/62 Batch time: 0.2703
Phase: train Epoch: 1/1 Iter: 41/62 Batch time: 0.2720
Phase: train Epoch: 1/1 Iter: 42/62 Batch time: 0.2771
Phase: train Epoch: 1/1 Iter: 43/62 Batch time: 0.2775
Phase: train Epoch: 1/1 Iter: 44/62 Batch time: 0.2670
Phase: train Epoch: 1/1 Iter: 45/62 Batch time: 0.2757
Phase: train Epoch: 1/1 Iter: 46/62 Batch time: 0.2673
Phase: train Epoch: 1/1 Iter: 47/62 Batch time: 0.2774
Phase: train Epoch: 1/1 Iter: 48/62 Batch time: 0.2778
Phase: train Epoch: 1/1 Iter: 49/62 Batch time: 0.2753
Phase: train Epoch: 1/1 Iter: 50/62 Batch time: 0.2765
Phase: train Epoch: 1/1 Iter: 51/62 Batch time: 0.2756
Phase: train Epoch: 1/1 Iter: 52/62 Batch time: 0.2677
Phase: train Epoch: 1/1 Iter: 53/62 Batch time: 0.2698
Phase: train Epoch: 1/1 Iter: 54/62 Batch time: 0.2776
Phase: train Epoch: 1/1 Iter: 55/62 Batch time: 0.2731
Phase: train Epoch: 1/1 Iter: 56/62 Batch time: 0.2695
Phase: train Epoch: 1/1 Iter: 57/62 Batch time: 0.2705
Phase: train Epoch: 1/1 Iter: 58/62 Batch time: 0.2717
Phase: train Epoch: 1/1 Iter: 59/62 Batch time: 0.2735
Phase: train Epoch: 1/1 Iter: 60/62 Batch time: 0.2804
Phase: train Epoch: 1/1 Iter: 61/62 Batch time: 0.2787
Phase: train Epoch: 1/1 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/1 Iter: 1/39 Batch time: 0.2078
Phase: validation Epoch: 1/1 Iter: 2/39 Batch time: 0.2002
Phase: validation Epoch: 1/1 Iter: 3/39 Batch time: 0.2067
Phase: validation Epoch: 1/1 Iter: 4/39 Batch time: 0.2100
Phase: validation Epoch: 1/1 Iter: 5/39 Batch time: 0.2090
Phase: validation Epoch: 1/1 Iter: 6/39 Batch time: 0.2075
Phase: validation Epoch: 1/1 Iter: 7/39 Batch time: 0.2005
Phase: validation Epoch: 1/1 Iter: 8/39 Batch time: 0.2022
Phase: validation Epoch: 1/1 Iter: 9/39 Batch time: 0.2024
Phase: validation Epoch: 1/1 Iter: 10/39 Batch time: 0.2056
Phase: validation Epoch: 1/1 Iter: 11/39 Batch time: 0.2000
Phase: validation Epoch: 1/1 Iter: 12/39 Batch time: 0.1978
Phase: validation Epoch: 1/1 Iter: 13/39 Batch time: 0.2025
Phase: validation Epoch: 1/1 Iter: 14/39 Batch time: 0.2023
Phase: validation Epoch: 1/1 Iter: 15/39 Batch time: 0.2052
Phase: validation Epoch: 1/1 Iter: 16/39 Batch time: 0.2087
Phase: validation Epoch: 1/1 Iter: 17/39 Batch time: 0.2075
Phase: validation Epoch: 1/1 Iter: 18/39 Batch time: 0.2064
Phase: validation Epoch: 1/1 Iter: 19/39 Batch time: 0.2074
Phase: validation Epoch: 1/1 Iter: 20/39 Batch time: 0.2083
Phase: validation Epoch: 1/1 Iter: 21/39 Batch time: 0.2076
Phase: validation Epoch: 1/1 Iter: 22/39 Batch time: 0.2053
Phase: validation Epoch: 1/1 Iter: 23/39 Batch time: 0.2084
Phase: validation Epoch: 1/1 Iter: 24/39 Batch time: 0.2086
Phase: validation Epoch: 1/1 Iter: 25/39 Batch time: 0.2114
Phase: validation Epoch: 1/1 Iter: 26/39 Batch time: 0.2129
Phase: validation Epoch: 1/1 Iter: 27/39 Batch time: 0.2086
Phase: validation Epoch: 1/1 Iter: 28/39 Batch time: 0.2117
Phase: validation Epoch: 1/1 Iter: 29/39 Batch time: 0.2155
Phase: validation Epoch: 1/1 Iter: 30/39 Batch time: 0.2056
Phase: validation Epoch: 1/1 Iter: 31/39 Batch time: 0.2103
Phase: validation Epoch: 1/1 Iter: 32/39 Batch time: 0.2063
Phase: validation Epoch: 1/1 Iter: 33/39 Batch time: 0.2074
Phase: validation Epoch: 1/1 Iter: 34/39 Batch time: 0.2099
Phase: validation Epoch: 1/1 Iter: 35/39 Batch time: 0.2169
Phase: validation Epoch: 1/1 Iter: 36/39 Batch time: 0.2041
Phase: validation Epoch: 1/1 Iter: 37/39 Batch time: 0.2063
Phase: validation Epoch: 1/1 Iter: 38/39 Batch time: 0.2109
Phase: validation Epoch: 1/1 Iter: 39/39 Batch time: 0.0707
Phase: validation   Epoch: 1/1 Loss: 0.6432 Acc: 0.6536
Training completed in 0m 28s
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
 13%|#2        | 5.73M/44.7M [00:00<00:00, 59.9MB/s]
 27%|##6       | 12.0M/44.7M [00:00<00:00, 63.3MB/s]
 71%|#######1  | 31.9M/44.7M [00:00<00:00, 130MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 129MB/s]
Training started:
Phase: train Epoch: 1/1 Iter: 1/62 Batch time: 0.2622
Phase: train Epoch: 1/1 Iter: 2/62 Batch time: 0.2360
Phase: train Epoch: 1/1 Iter: 3/62 Batch time: 0.2358
Phase: train Epoch: 1/1 Iter: 4/62 Batch time: 0.2520
Phase: train Epoch: 1/1 Iter: 5/62 Batch time: 0.2293
Phase: train Epoch: 1/1 Iter: 6/62 Batch time: 0.2274
Phase: train Epoch: 1/1 Iter: 7/62 Batch time: 0.2261
Phase: train Epoch: 1/1 Iter: 8/62 Batch time: 0.2393
Phase: train Epoch: 1/1 Iter: 9/62 Batch time: 0.2268
Phase: train Epoch: 1/1 Iter: 10/62 Batch time: 0.2241
Phase: train Epoch: 1/1 Iter: 11/62 Batch time: 0.2336
Phase: train Epoch: 1/1 Iter: 12/62 Batch time: 0.2253
Phase: train Epoch: 1/1 Iter: 13/62 Batch time: 0.2327
Phase: train Epoch: 1/1 Iter: 14/62 Batch time: 0.2255
Phase: train Epoch: 1/1 Iter: 15/62 Batch time: 0.2335
Phase: train Epoch: 1/1 Iter: 16/62 Batch time: 0.2322
Phase: train Epoch: 1/1 Iter: 17/62 Batch time: 0.2317
Phase: train Epoch: 1/1 Iter: 18/62 Batch time: 0.2398
Phase: train Epoch: 1/1 Iter: 19/62 Batch time: 0.2314
Phase: train Epoch: 1/1 Iter: 20/62 Batch time: 0.2356
Phase: train Epoch: 1/1 Iter: 21/62 Batch time: 0.2256
Phase: train Epoch: 1/1 Iter: 22/62 Batch time: 0.2283
Phase: train Epoch: 1/1 Iter: 23/62 Batch time: 0.2353
Phase: train Epoch: 1/1 Iter: 24/62 Batch time: 0.2382
Phase: train Epoch: 1/1 Iter: 25/62 Batch time: 0.2296
Phase: train Epoch: 1/1 Iter: 26/62 Batch time: 0.2328
Phase: train Epoch: 1/1 Iter: 27/62 Batch time: 0.2267
Phase: train Epoch: 1/1 Iter: 28/62 Batch time: 0.2394
Phase: train Epoch: 1/1 Iter: 29/62 Batch time: 0.2256
Phase: train Epoch: 1/1 Iter: 30/62 Batch time: 0.2294
Phase: train Epoch: 1/1 Iter: 31/62 Batch time: 0.2348
Phase: train Epoch: 1/1 Iter: 32/62 Batch time: 0.2719
Phase: train Epoch: 1/1 Iter: 33/62 Batch time: 0.2379
Phase: train Epoch: 1/1 Iter: 34/62 Batch time: 0.2297
Phase: train Epoch: 1/1 Iter: 35/62 Batch time: 0.2308
Phase: train Epoch: 1/1 Iter: 36/62 Batch time: 0.2312
Phase: train Epoch: 1/1 Iter: 37/62 Batch time: 0.2317
Phase: train Epoch: 1/1 Iter: 38/62 Batch time: 0.2364
Phase: train Epoch: 1/1 Iter: 39/62 Batch time: 0.2372
Phase: train Epoch: 1/1 Iter: 40/62 Batch time: 0.2231
Phase: train Epoch: 1/1 Iter: 41/62 Batch time: 0.2207
Phase: train Epoch: 1/1 Iter: 42/62 Batch time: 0.2465
Phase: train Epoch: 1/1 Iter: 43/62 Batch time: 0.2248
Phase: train Epoch: 1/1 Iter: 44/62 Batch time: 0.2223
Phase: train Epoch: 1/1 Iter: 45/62 Batch time: 0.2277
Phase: train Epoch: 1/1 Iter: 46/62 Batch time: 0.2307
Phase: train Epoch: 1/1 Iter: 47/62 Batch time: 0.2345
Phase: train Epoch: 1/1 Iter: 48/62 Batch time: 0.2187
Phase: train Epoch: 1/1 Iter: 49/62 Batch time: 0.2196
Phase: train Epoch: 1/1 Iter: 50/62 Batch time: 0.2266
Phase: train Epoch: 1/1 Iter: 51/62 Batch time: 0.2256
Phase: train Epoch: 1/1 Iter: 52/62 Batch time: 0.2244
Phase: train Epoch: 1/1 Iter: 53/62 Batch time: 0.2250
Phase: train Epoch: 1/1 Iter: 54/62 Batch time: 0.2248
Phase: train Epoch: 1/1 Iter: 55/62 Batch time: 0.2275
Phase: train Epoch: 1/1 Iter: 56/62 Batch time: 0.2242
Phase: train Epoch: 1/1 Iter: 57/62 Batch time: 0.2327
Phase: train Epoch: 1/1 Iter: 58/62 Batch time: 0.2266
Phase: train Epoch: 1/1 Iter: 59/62 Batch time: 0.2207
Phase: train Epoch: 1/1 Iter: 60/62 Batch time: 0.2227
Phase: train Epoch: 1/1 Iter: 61/62 Batch time: 0.2184
Phase: train Epoch: 1/1 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/1 Iter: 1/39 Batch time: 0.1768
Phase: validation Epoch: 1/1 Iter: 2/39 Batch time: 0.1652
Phase: validation Epoch: 1/1 Iter: 3/39 Batch time: 0.1643
Phase: validation Epoch: 1/1 Iter: 4/39 Batch time: 0.1684
Phase: validation Epoch: 1/1 Iter: 5/39 Batch time: 0.1699
Phase: validation Epoch: 1/1 Iter: 6/39 Batch time: 0.1638
Phase: validation Epoch: 1/1 Iter: 7/39 Batch time: 0.1678
Phase: validation Epoch: 1/1 Iter: 8/39 Batch time: 0.1625
Phase: validation Epoch: 1/1 Iter: 9/39 Batch time: 0.1630
Phase: validation Epoch: 1/1 Iter: 10/39 Batch time: 0.1615
Phase: validation Epoch: 1/1 Iter: 11/39 Batch time: 0.1723
Phase: validation Epoch: 1/1 Iter: 12/39 Batch time: 0.1650
Phase: validation Epoch: 1/1 Iter: 13/39 Batch time: 0.1640
Phase: validation Epoch: 1/1 Iter: 14/39 Batch time: 0.1656
Phase: validation Epoch: 1/1 Iter: 15/39 Batch time: 0.1607
Phase: validation Epoch: 1/1 Iter: 16/39 Batch time: 0.1683
Phase: validation Epoch: 1/1 Iter: 17/39 Batch time: 0.1712
Phase: validation Epoch: 1/1 Iter: 18/39 Batch time: 0.1651
Phase: validation Epoch: 1/1 Iter: 19/39 Batch time: 0.1630
Phase: validation Epoch: 1/1 Iter: 20/39 Batch time: 0.1671
Phase: validation Epoch: 1/1 Iter: 21/39 Batch time: 0.1688
Phase: validation Epoch: 1/1 Iter: 22/39 Batch time: 0.1638
Phase: validation Epoch: 1/1 Iter: 23/39 Batch time: 0.1645
Phase: validation Epoch: 1/1 Iter: 24/39 Batch time: 0.1646
Phase: validation Epoch: 1/1 Iter: 25/39 Batch time: 0.1629
Phase: validation Epoch: 1/1 Iter: 26/39 Batch time: 0.1630
Phase: validation Epoch: 1/1 Iter: 27/39 Batch time: 0.1610
Phase: validation Epoch: 1/1 Iter: 28/39 Batch time: 0.1708
Phase: validation Epoch: 1/1 Iter: 29/39 Batch time: 0.1704
Phase: validation Epoch: 1/1 Iter: 30/39 Batch time: 0.1630
Phase: validation Epoch: 1/1 Iter: 31/39 Batch time: 0.1629
Phase: validation Epoch: 1/1 Iter: 32/39 Batch time: 0.1676
Phase: validation Epoch: 1/1 Iter: 33/39 Batch time: 0.1614
Phase: validation Epoch: 1/1 Iter: 34/39 Batch time: 0.1697
Phase: validation Epoch: 1/1 Iter: 35/39 Batch time: 0.1612
Phase: validation Epoch: 1/1 Iter: 36/39 Batch time: 0.1708
Phase: validation Epoch: 1/1 Iter: 37/39 Batch time: 0.1616
Phase: validation Epoch: 1/1 Iter: 38/39 Batch time: 0.1682
Phase: validation Epoch: 1/1 Iter: 39/39 Batch time: 0.0559
Phase: validation   Epoch: 1/1 Loss: 0.6432 Acc: 0.6536
Training completed in 0m 24s
 </code>
 </pre>
 </details>

---

## 5. tutorial_QGAN.html <a name="demo4"></a>

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

## 6. tutorial_kernels_module.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_kernels_module.html):

```
The kernel value between the first and second datapoint is 0.093
[[1.    0.093 0.012 0.721 0.149 0.055]
 [0.093 1.    0.056 0.218 0.73  0.213]
 [0.012 0.056 1.    0.032 0.191 0.648]
 [0.721 0.218 0.032 1.    0.391 0.226]
 [0.149 0.73  0.191 0.391 1.    0.509]
 [0.055 0.213 0.648 0.226 0.509 1.   ]]
The accuracy of the kernel with random parameters is 0.833
The kernel-target alignment for our dataset and random parameters is 0.081
Step 50 - Alignment = 0.098
Step 100 - Alignment = 0.121
Step 150 - Alignment = 0.141
Step 200 - Alignment = 0.173
Step 250 - Alignment = 0.196
Step 300 - Alignment = 0.224
Step 350 - Alignment = 0.245
Step 400 - Alignment = 0.261
Step 450 - Alignment = 0.276
Step 500 - Alignment = 0.289
The accuracy of a kernel with trained parameters is 1.000
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_kernels_module.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
The kernel value between the first and second datapoint is 0.093
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
[[1.    0.093 0.012 0.721 0.149 0.055]
 [0.093 1.    0.056 0.218 0.73  0.213]
 [0.012 0.056 1.    0.032 0.191 0.648]
 [0.721 0.218 0.032 1.    0.391 0.226]
 [0.149 0.73  0.191 0.391 1.    0.509]
 [0.055 0.213 0.648 0.226 0.509 1.   ]]
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
The accuracy of the kernel with random parameters is 0.833
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
The kernel-target alignment for our dataset and random parameters is 0.081
/home/runner/work/qml/qml/demonstrations/tutorial_kernels_module.py:253: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Step 50 - Alignment = 0.098
Step 100 - Alignment = 0.121
Step 150 - Alignment = 0.141
Step 200 - Alignment = 0.173
```

---

## 7. tutorial_vqe_parallel.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.81
Evaluation time: 334.56 s
Evaluation time: 119.16 s
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.80
Evaluation time: 300.71 s
Evaluation time: 107.33 s
```

---

## 8. tutorial_adaptive_circuits.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.01278217515759826
Excitation : [0, 1, 2, 5], Gradient: -2.0328790734103208e-20
Excitation : [0, 1, 2, 7], Gradient: -1.0842021724855052e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170160119
Excitation : [0, 1, 3, 4], Gradient: 6.09863722023096e-20
Excitation : [0, 1, 3, 6], Gradient: -1.0842021724855059e-19
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170160133
Excitation : [0, 1, 4, 5], Gradient: -0.023581529020658853
Excitation : [0, 1, 4, 7], Gradient: 0.0
Excitation : [0, 1, 4, 9], Gradient: 0.0
Excitation : [0, 1, 5, 6], Gradient: 0.0
Excitation : [0, 1, 5, 8], Gradient: -6.098637220230905e-20
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020658853
Excitation : [0, 1, 6, 9], Gradient: 0.0
Excitation : [0, 1, 7, 8], Gradient: 1.3552527156068854e-19
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485602746
[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 8], [0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 8, 9]]
Excitation : [0, 2], Gradient: -0.005062536239409636
Excitation : [0, 4], Gradient: 2.0325203781891008e-17
Excitation : [0, 6], Gradient: -1.0219994129978932e-18
Excitation : [0, 8], Gradient: -0.0009448044625863393
Excitation : [1, 3], Gradient: 0.0049266168770772285
Excitation : [1, 5], Gradient: 7.343869440637225e-18
Excitation : [1, 7], Gradient: -2.7844571783256364e-18
Excitation : [1, 9], Gradient: 0.0014535534854193673
[[0, 2], [0, 8], [1, 3], [1, 9]]
n = 0,  E = -7.86266587 H, t = 1.71 s
n = 1,  E = -7.87094621 H, t = 2.20 s
n = 2,  E = -7.87563100 H, t = 1.69 s
n = 3,  E = -7.87829146 H, t = 2.21 s
n = 4,  E = -7.87981705 H, t = 1.69 s
n = 5,  E = -7.88070477 H, t = 2.19 s
n = 6,  E = -7.88123143 H, t = 1.72 s
n = 7,  E = -7.88155161 H, t = 2.21 s
n = 8,  E = -7.88175217 H, t = 1.74 s
n = 9,  E = -7.88188237 H, t = 2.19 s
n = 10,  E = -7.88197041 H, t = 2.25 s
n = 11,  E = -7.88203267 H, t = 1.71 s
n = 12,  E = -7.88207879 H, t = 2.22 s
n = 13,  E = -7.88211452 H, t = 1.71 s
n = 14,  E = -7.88214335 H, t = 2.23 s
n = 15,  E = -7.88216743 H, t = 1.73 s
n = 16,  E = -7.88218814 H, t = 2.24 s
n = 17,  E = -7.88220634 H, t = 1.72 s
n = 18,  E = -7.88222261 H, t = 2.23 s
n = 19,  E = -7.88223734 H, t = 1.71 s
<1024x1024 sparse matrix of type '<class 'numpy.complex128'>'
    with 11264 stored elements in COOrdinate format>
n = 0,  E = -7.86266587 H, t = 0.10 s
n = 1,  E = -7.87094621 H, t = 0.10 s
n = 2,  E = -7.87563100 H, t = 0.10 s
n = 3,  E = -7.87829146 H, t = 0.10 s
n = 4,  E = -7.87981705 H, t = 0.10 s
n = 5,  E = -7.88070477 H, t = 0.10 s
n = 6,  E = -7.88123143 H, t = 0.10 s
n = 7,  E = -7.88155161 H, t = 0.10 s
n = 8,  E = -7.88175217 H, t = 0.10 s
n = 9,  E = -7.88188237 H, t = 0.10 s
n = 10,  E = -7.88197041 H, t = 0.10 s
n = 11,  E = -7.88203267 H, t = 0.10 s
n = 12,  E = -7.88207879 H, t = 0.10 s
n = 13,  E = -7.88211452 H, t = 0.10 s
n = 14,  E = -7.88214335 H, t = 0.10 s
n = 15,  E = -7.88216743 H, t = 0.10 s
n = 16,  E = -7.88218814 H, t = 0.10 s
n = 17,  E = -7.88220634 H, t = 0.10 s
n = 18,  E = -7.88222261 H, t = 0.10 s
n = 19,  E = -7.88223734 H, t = 0.09 s
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
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157603867
Excitation : [0, 1, 2, 5], Gradient: -6.776263578034379e-21
Excitation : [0, 1, 2, 7], Gradient: 1.4907779871675684e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170160816
Excitation : [0, 1, 3, 4], Gradient: 7.453889935837851e-20
Excitation : [0, 1, 3, 6], Gradient: -6.776263578034404e-20
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170160823
Excitation : [0, 1, 4, 5], Gradient: -0.023581529020660436
Excitation : [0, 1, 4, 7], Gradient: 0.0
Excitation : [0, 1, 4, 9], Gradient: 0.0
Excitation : [0, 1, 5, 6], Gradient: 0.0
Excitation : [0, 1, 5, 8], Gradient: -1.0164395367051532e-19
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020660436
Excitation : [0, 1, 6, 9], Gradient: 0.0
Excitation : [0, 1, 7, 8], Gradient: -9.486769009248113e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485602415
[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 8], [0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 8, 9]]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Excitation : [0, 2], Gradient: -0.005062536239403361
Excitation : [0, 4], Gradient: -1.1783474063767686e-19
Excitation : [0, 6], Gradient: -1.2295606776308945e-19
Excitation : [0, 8], Gradient: -0.0009448044625853639
Excitation : [1, 3], Gradient: 0.004926616877071086
Excitation : [1, 5], Gradient: -2.0166231293011255e-19
Excitation : [1, 7], Gradient: 6.22777104414388e-19
Excitation : [1, 9], Gradient: 0.0014535534854178628
[[0, 2], [0, 8], [1, 3], [1, 9]]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
n = 0,  E = -7.86266587 H, t = 2.49 s
n = 1,  E = -7.87094621 H, t = 2.95 s
n = 2,  E = -7.87563100 H, t = 2.97 s
n = 3,  E = -7.87829146 H, t = 2.54 s
n = 4,  E = -7.87981705 H, t = 2.90 s
n = 5,  E = -7.88070477 H, t = 3.02 s
n = 6,  E = -7.88123143 H, t = 2.42 s
n = 7,  E = -7.88155161 H, t = 2.83 s
n = 8,  E = -7.88175217 H, t = 2.87 s
n = 9,  E = -7.88188237 H, t = 2.38 s
n = 10,  E = -7.88197041 H, t = 2.85 s
n = 11,  E = -7.88203267 H, t = 2.87 s
n = 12,  E = -7.88207879 H, t = 2.39 s
n = 13,  E = -7.88211452 H, t = 2.81 s
n = 14,  E = -7.88214335 H, t = 2.86 s
n = 15,  E = -7.88216743 H, t = 2.38 s
n = 16,  E = -7.88218814 H, t = 2.86 s
n = 17,  E = -7.88220634 H, t = 2.90 s
n = 18,  E = -7.88222261 H, t = 2.47 s
n = 19,  E = -7.88223734 H, t = 2.82 s
<1024x1024 sparse matrix of type '<class 'numpy.complex128'>'
    with 11264 stored elements in COOrdinate format>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
n = 0,  E = -7.86266587 H, t = 0.13 s
n = 1,  E = -7.87094621 H, t = 0.13 s
n = 2,  E = -7.87563100 H, t = 0.13 s
n = 3,  E = -7.87829146 H, t = 0.13 s
n = 4,  E = -7.87981705 H, t = 0.13 s
n = 5,  E = -7.88070477 H, t = 0.13 s
n = 6,  E = -7.88123143 H, t = 0.13 s
n = 7,  E = -7.88155161 H, t = 0.13 s
n = 8,  E = -7.88175217 H, t = 0.13 s
n = 9,  E = -7.88188237 H, t = 0.13 s
n = 10,  E = -7.88197041 H, t = 0.13 s
n = 11,  E = -7.88203267 H, t = 0.12 s
n = 12,  E = -7.88207879 H, t = 0.13 s
 </code>
 </pre>
 </details>

---

## 9. tutorial_qaoa_intro.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
0: ──H──────RZ(1)──H──H──╭RZ(0.5)──H──H──────RZ(1)──H──H──╭RZ(0.5)──H──┤ ⟨Z⟩
1: ──RZ(1)──H────────────╰RZ(0.5)──H──RZ(1)──H────────────╰RZ(0.5)──H──┤ ⟨Z⟩
0: ──RX(0.5)──╭C──┤ ⟨Z⟩
1: ──H────────╰X──┤ ⟨Z⟩
0: ──RX(0.3)──╭C──RX(0.4)──╭C──RX(0.5)──╭C──┤ ⟨Z⟩
1: ──H────────╰X──H────────╰X──H────────╰X──┤ ⟨Z⟩
Cost Hamiltonian   (-0.25) [Z3]
+ (0.5) [Z0]
+ (0.5) [Z1]
+ (1.25) [Z2]
+ (0.75) [Z0 Z1]
+ (0.75) [Z0 Z2]
+ (0.75) [Z1 Z2]
+ (0.75) [Z2 Z3]
Mixer Hamiltonian   (1) [X0]
+ (1) [X1]
+ (1) [X2]
+ (1) [X3]
Optimal Parameters
[[0.5980635175924566, 0.9419848542526791], [0.5279728111755442, 0.855528453707565]]
Optimal Parameters
[[0.45959941488399797, 0.9609527141073113], [0.2702996191454587, 0.7804239603322595]]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qaoa_intro.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/operation.py:730: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
 0: ──H──────RZ(1)──H──H──╭RZ(0.5)──H──H──────RZ(1)──H──H──╭RZ(0.5)──H──┤ ⟨Z⟩
 1: ──RZ(1)──H────────────╰RZ(0.5)──H──RZ(1)──H────────────╰RZ(0.5)──H──┤ ⟨Z⟩
0: ──RX(0.5)──╭C──┤ ⟨Z⟩
1: ──H────────╰X──┤ ⟨Z⟩
/home/runner/work/qml/qml/demonstrations/tutorial_qaoa_intro.py:160: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
 0: ──RX(0.3)──╭C──RX(0.4)──╭C──RX(0.5)──╭C──┤ ⟨Z⟩
 1: ──H────────╰X──H────────╰X──H────────╰X──┤ ⟨Z⟩
Cost Hamiltonian   (-0.25) [Z3]
+ (0.5) [Z0]
+ (0.5) [Z1]
+ (1.25) [Z2]
+ (0.75) [Z0 Z1]
+ (0.75) [Z0 Z2]
+ (0.75) [Z1 Z2]
+ (0.75) [Z2 Z3]
Mixer Hamiltonian   (1) [X0]
+ (1) [X1]
+ (1) [X2]
+ (1) [X3]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/home/runner/work/qml/qml/demonstrations/tutorial_qaoa_intro.py:305: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
 </code>
 </pre>
 </details>

---

## 10. tutorial_gaussian_transformation.html <a name="demo9"></a>

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

## 11. tutorial_gbs.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_gbs.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_gbs.py:165: UserWarning: 'Interferometer' is deprecated and will be renamed 'InterferometerUnitary'
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

## 12. tutorial_pasqal.html <a name="demo11"></a>

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

## 13. tutorial_variational_classifier.html <a name="demo12"></a>

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

## 14. tutorial_backprop.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.11971365706871569
0: ──RX(0.375)──╭C─────────────────╭X──RX(0.599)──╭C──────╭X──╭┤ ⟨Y ⊗ Z⟩
1: ──RY(0.951)──╰X──╭C──RY(0.156)──│──────────────╰X──╭C──│───│┤
2: ──RZ(0.732)──────╰X─────────────╰C──RZ(0.156)──────╰X──╰C──╰┤ ⟨Y ⊗ Z⟩
-0.0651887722495813
[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
 -7.61067572e-01  8.32667268e-17]
-0.0651887722495813
[[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
  -7.61067572e-01  8.32667268e-17]]
180
0.8947771876917632
Forward pass (best of 3): 0.006759593100014172 sec per loop
Gradient computation (best of 3): 2.5449242068999864 sec per loop
2.433453516005102
0.9358535378025419
Forward pass (best of 3): 0.04379045880000376 sec per loop
Backward pass (best of 3): 0.08781954410001162 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:72: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:78: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Expectation value: -0.11971365706871569
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:72: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:78: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
 0: ──RX(0.375)──╭C─────────────────╭X──RX(0.599)──╭C──────╭X──╭┤ ⟨Y ⊗ Z⟩
 1: ──RY(0.951)──╰X──╭C──RY(0.156)──│──────────────╰X──╭C──│───│┤
 2: ──RZ(0.732)──────╰X─────────────╰C──RZ(0.156)──────╰X──╰C──╰┤ ⟨Y ⊗ Z⟩
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:72: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:78: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
-0.0651887722495813
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:72: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:78: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
 -7.61067572e-01  8.32667268e-17]
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:72: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_backprop.py:78: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
-0.0651887722495813
```

---

## 15. tutorial_doubly_stochastic.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_doubly_stochastic.html):

```
Vanilla gradient descent min energy =  -4.605247234069292
Stochastic gradient descent (shots=100) min energy =  -4.60065517691614
Stochastic gradient descent (shots=1) min energy =  -4.457668962761634
Doubly stochastic gradient descent min energy =  -4.4990195930951575
Adaptive QSGD min energy =  -4.592548741613157
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_doubly_stochastic.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_doubly_stochastic.py:158: UserWarning: The init module will be deprecated soon, since templates can now provide a method that returns the shape of parameter tensors.
Vanilla gradient descent min energy =  -4.605247234069292
Stochastic gradient descent (shots=100) min energy =  -4.600655176916144
Stochastic gradient descent (shots=1) min energy =  -4.457668962761634
Doubly stochastic gradient descent min energy =  -4.4990195930951575
```

---

## 16. tutorial_mol_geo_opt.html <a name="demo15"></a>

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

## 17. tutorial_noisy_circuits.html <a name="demo16"></a>

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

## 18. tutorial_data_reuploading_classifier.html <a name="demo17"></a>

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

## 19. tutorial_vqe_spin_sectors.html <a name="demo18"></a>

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

## 20. tutorial_expressivity_fourier_series.html <a name="demo19"></a>

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

## 21. tutorial_multiclass_classification.html <a name="demo20"></a>

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
Iter:    29 | Cost: 0.0748658 | Acc train: 0.6785714 | Acc test: 0.7105263
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
Iter:    83 | Cost: 0.0823876 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    87 | Cost: 0.0150624 | Acc train: 0.9375000 | Acc test: 0.9210526
Iter:    91 | Cost: 0.0351690 | Acc train: 0.9107143 | Acc test: 0.9210526
Iter:    92 | Cost: 0.0555153 | Acc train: 0.9017857 | Acc test: 0.9210526
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
Iter:    29 | Cost: 0.0748657 | Acc train: 0.6785714 | Acc test: 0.7105263
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
Iter:    83 | Cost: 0.0823874 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    87 | Cost: 0.0150623 | Acc train: 0.9375000 | Acc test: 0.9210526
Iter:    91 | Cost: 0.0351689 | Acc train: 0.9107143 | Acc test: 0.9210526
Iter:    92 | Cost: 0.0555152 | Acc train: 0.9017857 | Acc test: 0.9210526
Iter:    95 | Cost: 0.0358330 | Acc train: 0.8482143 | Acc test: 0.8947368
Iter:    97 | Cost: 0.0946534 | Acc train: 0.8035714 | Acc test: 0.8947368
Iter:    98 | Cost: 0.0701063 | Acc train: 0.8839286 | Acc test: 0.8684211
Iter:    99 | Cost: 0.0827179 | Acc train: 0.9642857 | Acc test: 0.9473684
 </code>
 </pre>
 </details>

---

## 22. tutorial_vqe_qng.html <a name="demo21"></a>

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
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/transforms/metric_tensor.py:164: UserWarning: The keyword argument diag_approx is deprecated. Please use approx='diag' instead.
Iteration = 0,  Energy = 0.51052556 Ha,  Convergence parameter = 0.06664604 Ha
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:195: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
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
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/transforms/metric_tensor.py:164: UserWarning: The keyword argument diag_approx is deprecated. Please use approx='diag' instead.
Iteration = 0,  Energy = -0.32164518 Ha
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:195: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
Iteration = 4,  Energy = -0.46875033 Ha
Iteration = 8,  Energy = -0.85091055 Ha
Iteration = 12,  Energy = -1.13575339 Ha
Iteration = 16,  Energy = -1.13618916 Ha
Final convergence parameter = 0.00000022 Ha
Number of iterations =  17
 </code>
 </pre>
 </details>

---

## 23. tutorial_general_parshift.html <a name="demo22"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_general_parshift.html):

```
For 2 qubits the spectrum is [-2.0, -1.0, 0.0, 1.0, 2.0].
For 4 qubits the spectrum is [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0].
For 5 qubits the spectrum is [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0].
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_general_parshift.html):

```
For 2 qubits the spectrum is [-2.0, -1.0, 0, 1.0, 2.0].
For 4 qubits the spectrum is [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0].
For 5 qubits the spectrum is [-5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0].
```

---

## 24. tutorial_qnn_module_tf.html <a name="demo23"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 7s - loss: 0.3931 - accuracy: 0.7067 - val_loss: 0.2683 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.2107 - accuracy: 0.8600 - val_loss: 0.1992 - val_accuracy: 0.8200
30/30 - 7s - loss: 0.1670 - accuracy: 0.8800 - val_loss: 0.1854 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.1602 - accuracy: 0.8800 - val_loss: 0.1732 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.1514 - accuracy: 0.8800 - val_loss: 0.1692 - val_accuracy: 0.8600
30/30 - 7s - loss: 0.1433 - accuracy: 0.8800 - val_loss: 0.1787 - val_accuracy: 0.8200
30/30 - 14s - loss: 0.4068 - accuracy: 0.6600 - val_loss: 0.3008 - val_accuracy: 0.7400
30/30 - 14s - loss: 0.2845 - accuracy: 0.7733 - val_loss: 0.2298 - val_accuracy: 0.8200
30/30 - 14s - loss: 0.2180 - accuracy: 0.8067 - val_loss: 0.1976 - val_accuracy: 0.8200
30/30 - 14s - loss: 0.1904 - accuracy: 0.8533 - val_loss: 0.1809 - val_accuracy: 0.8200
30/30 - 14s - loss: 0.1702 - accuracy: 0.8600 - val_loss: 0.1719 - val_accuracy: 0.8600
30/30 - 14s - loss: 0.1538 - accuracy: 0.8600 - val_loss: 0.1862 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 8s - loss: 0.3931 - accuracy: 0.7067 - val_loss: 0.2683 - val_accuracy: 0.8600
30/30 - 8s - loss: 0.2107 - accuracy: 0.8600 - val_loss: 0.1992 - val_accuracy: 0.8200
30/30 - 8s - loss: 0.1670 - accuracy: 0.8800 - val_loss: 0.1854 - val_accuracy: 0.8600
30/30 - 8s - loss: 0.1602 - accuracy: 0.8800 - val_loss: 0.1732 - val_accuracy: 0.8600
30/30 - 8s - loss: 0.1514 - accuracy: 0.8800 - val_loss: 0.1692 - val_accuracy: 0.8600
30/30 - 8s - loss: 0.1433 - accuracy: 0.8800 - val_loss: 0.1787 - val_accuracy: 0.8200
30/30 - 16s - loss: 0.4068 - accuracy: 0.6600 - val_loss: 0.3008 - val_accuracy: 0.7400
30/30 - 16s - loss: 0.2845 - accuracy: 0.7733 - val_loss: 0.2298 - val_accuracy: 0.8200
30/30 - 16s - loss: 0.2180 - accuracy: 0.8067 - val_loss: 0.1976 - val_accuracy: 0.8200
30/30 - 16s - loss: 0.1904 - accuracy: 0.8533 - val_loss: 0.1809 - val_accuracy: 0.8200
30/30 - 16s - loss: 0.1702 - accuracy: 0.8600 - val_loss: 0.1719 - val_accuracy: 0.8600
30/30 - 16s - loss: 0.1538 - accuracy: 0.8600 - val_loss: 0.1862 - val_accuracy: 0.8400
```

---

## 25. tutorial_vqt.html <a name="demo24"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqt.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 0: ──X──────RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C──────────────────────╭RX(1)──╭┤ ⟨H0⟩
 1: ──RZ(1)──RY(1)──RX(1)─────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C──────────────│───────├┤ ⟨H0⟩
 2: ──X──────RZ(1)──RY(1)──RX(1)──────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────│───────├┤ ⟨H0⟩
 3: ──RZ(1)──RY(1)──RX(1)─────────────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)──╰C──────╰┤ ⟨H0⟩
H0 =
[[ 4.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j -4.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j]
 [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j -4.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  4.+0.j]]
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
/home/runner/work/qml/qml/demonstrations/tutorial_vqt.py:269: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
 0: ──X──────RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C──────────────────────╭RX(1)──╭┤ ⟨H0⟩
 1: ──RZ(1)──RY(1)──RX(1)─────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C──────────────│───────├┤ ⟨H0⟩
 2: ──X──────RZ(1)──RY(1)──RX(1)──────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────│───────├┤ ⟨H0⟩
 3: ──RZ(1)──RY(1)──RX(1)─────────────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)──╰C──────╰┤ ⟨H0⟩
H0 =
[[ 4.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j -4.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j]
 [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j -4.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  4.+0.j]]
/home/runner/work/qml/qml/demonstrations/tutorial_vqt.py:269: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
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
 </code>
 </pre>
 </details>

---

## 26. tutorial_vqe.html <a name="demo25"></a>

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

## 27. tutorial_local_cost_functions.html <a name="demo26"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_local_cost_functions.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost after step     5:  1.0000000
Cost after step    10:  1.0000000
Cost after step    15:  1.0000000
Cost after step    20:  1.0000000
Cost after step    25:  1.0000000
Cost after step    30:  1.0000000
Cost after step    35:  1.0000000
Cost after step    40:  1.0000000
Cost after step    45:  1.0000000
Cost after step    50:  1.0000000
Cost after step    55:  1.0000000
Cost after step    60:  1.0000000
Cost after step    65:  1.0000000
Cost after step    70:  1.0000000
Cost after step    75:  1.0000000
Cost after step    80:  1.0000000
Cost after step    85:  1.0000000
Cost after step    90:  1.0000000
Cost after step    95:  1.0000000
Cost after step   100:  1.0000000
 0: ──RX(3)──RY(0)──╭C──────────────────╭┤ Probs
 1: ──RX(3)──RY(0)──╰X──╭C──────────────├┤ Probs
 2: ──RX(3)──RY(0)──────╰X──╭C──────────├┤ Probs
 3: ──RX(3)──RY(0)──────────╰X──╭C──────├┤ Probs
 4: ──RX(3)──RY(0)──────────────╰X──╭C──├┤ Probs
 5: ──RX(3)──RY(0)──────────────────╰X──╰┤ Probs
Cost after step     5:  0.9871000
Cost after step    10:  0.9651000
Cost after step    15:  0.9173000
Cost after step    20:  0.8059000
Cost after step    25:  0.6213000
Cost after step    30:  0.3703000
Cost after step    35:  0.1821000
Cost after step    40:  0.0684000
 0: ──RX(0.44)──RY(-0.00321)──╭C──────────────────┤ Probs
 1: ──RX(3.01)──RY(-4e-05)────╰X──╭C──────────────┤
 2: ──RX(3)─────RY(0)─────────────╰X──╭C──────────┤
 3: ──RX(3)─────RY(0)─────────────────╰X──╭C──────┤
 4: ──RX(3)─────RY(0)─────────────────────╰X──╭C──┤
 5: ──RX(3)─────RY(0)─────────────────────────╰X──┤
tensor(1., requires_grad=True)
Current cost: 0.9999999999972213.
Initial cost: 0.9999999999999843.
Difference: 2.763012041384627e-12
0.9957
 0: ──RX(0.44)──RY(-0.00321)──╭C──────────────────╭┤ Probs
 1: ──RX(3.01)──RY(-4e-05)────╰X──╭C──────────────╰┤ Probs
 2: ──RX(3)─────RY(0)─────────────╰X──╭C───────────┤
 3: ──RX(3)─────RY(0)─────────────────╰X──╭C───────┤
 4: ──RX(3)─────RY(0)─────────────────────╰X──╭C───┤
 5: ──RX(3)─────RY(0)─────────────────────────╰X───┤
Cost after step    10:  0.9909000. Locality: 2
Cost after step    20:  0.9753000. Locality: 2
Cost after step    30:  0.9275000. Locality: 2
Cost after step    40:  0.8386000. Locality: 2
Cost after step    50:  0.6821000. Locality: 2
Cost after step    60:  0.4353000. Locality: 2
Cost after step    70:  0.2264000. Locality: 2
Cost after step    80:  0.0923000. Locality: 2
---Switching Locality---
Cost after step    90:  0.9901000. Locality: 3
Cost after step   100:  0.9737000. Locality: 3
Cost after step   110:  0.9400000. Locality: 3
Cost after step   120:  0.8711000. Locality: 3
Cost after step   130:  0.7228000. Locality: 3
Cost after step   140:  0.5156000. Locality: 3
Cost after step   150:  0.2846000. Locality: 3
Cost after step   160:  0.1285000. Locality: 3
---Switching Locality---
Cost after step   170:  0.9899000. Locality: 4
Cost after step   180:  0.9799000. Locality: 4
Cost after step   190:  0.9512000. Locality: 4
Cost after step   200:  0.8964000. Locality: 4
Cost after step   210:  0.7683000. Locality: 4
Cost after step   220:  0.5752000. Locality: 4
Cost after step   230:  0.3314000. Locality: 4
Cost after step   240:  0.1575000. Locality: 4
---Switching Locality---
Cost after step   250:  0.9942000. Locality: 5
Cost after step   260:  0.9866000. Locality: 5
Cost after step   270:  0.9641000. Locality: 5
Cost after step   280:  0.9120000. Locality: 5
Cost after step   290:  0.8136000. Locality: 5
Cost after step   300:  0.6380000. Locality: 5
Cost after step   310:  0.4004000. Locality: 5
Cost after step   320:  0.1996000. Locality: 5
---Switching Locality---
Cost after step   330:  0.9945000. Locality: 6
Cost after step   340:  0.9873000. Locality: 6
Cost after step   350:  0.9689000. Locality: 6
Cost after step   360:  0.9288000. Locality: 6
Cost after step   370:  0.8476000. Locality: 6
Cost after step   380:  0.6711000. Locality: 6
Cost after step   390:  0.4527000. Locality: 6
Cost after step   400:  0.2342000. Locality: 6
Cost after step   410:  0.1014000. Locality: 6
 0: ──RX(0.00069)──RY(-0.00297)──╭C──────────────────╭┤ Probs
 1: ──RX(0.00331)──RY(-0.0017)───╰X──╭C──────────────├┤ Probs
 2: ──RX(0.017)────RY(-0.00032)──────╰X──╭C──────────├┤ Probs
 3: ──RX(0.0496)───RY(4.5e-05)───────────╰X──╭C──────├┤ Probs
 4: ──RX(0.174)────RY(0.00258)───────────────╰X──╭C──├┤ Probs
 5: ──RX(0.599)────RY(0.0005)────────────────────╰X──╰┤ Probs
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
Cost after step   130:  0.2921000. Locality: 8
Trained:     5
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.7449000. Locality: 2
Cost after step    20:  0.5005000. Locality: 2
---Switching Locality---
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
---Switching Locality---
---Switching Locality---
Cost after step    50:  0.5099000. Locality: 7
---Switching Locality---
Cost after step    60:  0.6587000. Locality: 8
Cost after step    70:  0.4912000. Locality: 8
Cost after step    80:  0.2440000. Locality: 8
Trained:     6
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.8510000. Locality: 3
Cost after step    20:  0.7255000. Locality: 3
Cost after step    30:  0.5811000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6300000. Locality: 4
---Switching Locality---
Cost after step    50:  0.8801000. Locality: 5
Cost after step    60:  0.7395000. Locality: 5
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
Cost after step    20:  0.8358000. Locality: 2
Cost after step    30:  0.7614000. Locality: 2
Cost after step    40:  0.6718000. Locality: 2
Cost after step    50:  0.4921000. Locality: 2
---Switching Locality---
---Switching Locality---
Cost after step    60:  0.6433000. Locality: 4
---Switching Locality---
Cost after step    70:  0.7787000. Locality: 5
Cost after step    80:  0.5250000. Locality: 5
---Switching Locality---
Cost after step    90:  0.5270000. Locality: 6
---Switching Locality---
---Switching Locality---
Cost after step   100:  0.9690000. Locality: 8
Cost after step   110:  0.9106000. Locality: 8
Cost after step   120:  0.7561000. Locality: 8
Cost after step   130:  0.4537000. Locality: 8
Cost after step   140:  0.1197000. Locality: 8
Trained:     9
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.6759000. Locality: 2
---Switching Locality---
Cost after step    20:  0.8020000. Locality: 3
Cost after step    30:  0.6412000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6920000. Locality: 4
---Switching Locality---
Cost after step    50:  0.7176000. Locality: 5
Cost after step    60:  0.5973000. Locality: 5
Cost after step    70:  0.5469000. Locality: 6
Cost after step    80:  0.7019000. Locality: 7
Cost after step    90:  0.6006000. Locality: 7
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
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:241: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:247: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:241: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Cost after step     5:  1.0000000
Cost after step    10:  1.0000000
Cost after step    15:  1.0000000
Cost after step    20:  1.0000000
Cost after step    25:  1.0000000
Cost after step    30:  1.0000000
Cost after step    35:  1.0000000
Cost after step    40:  1.0000000
Cost after step    45:  1.0000000
Cost after step    50:  1.0000000
Cost after step    55:  1.0000000
Cost after step    60:  1.0000000
Cost after step    65:  1.0000000
Cost after step    70:  1.0000000
Cost after step    75:  1.0000000
Cost after step    80:  1.0000000
Cost after step    85:  1.0000000
Cost after step    90:  1.0000000
Cost after step    95:  1.0000000
Cost after step   100:  1.0000000
 0: ──RX(3)──RY(0)──╭C──────────────────╭┤ Probs
 1: ──RX(3)──RY(0)──╰X──╭C──────────────├┤ Probs
 2: ──RX(3)──RY(0)──────╰X──╭C──────────├┤ Probs
 3: ──RX(3)──RY(0)──────────╰X──╭C──────├┤ Probs
 4: ──RX(3)──RY(0)──────────────╰X──╭C──├┤ Probs
 5: ──RX(3)──RY(0)──────────────────╰X──╰┤ Probs
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:247: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Cost after step     5:  0.9871000
Cost after step    10:  0.9651000
Cost after step    15:  0.9173000
Cost after step    20:  0.8059000
Cost after step    25:  0.6213000
Cost after step    30:  0.3703000
Cost after step    35:  0.1821000
Cost after step    40:  0.0684000
 0: ──RX(0.44)──RY(-0.00321)──╭C──────────────────┤ Probs
 1: ──RX(3.01)──RY(-4e-05)────╰X──╭C──────────────┤
 2: ──RX(3)─────RY(0)─────────────╰X──╭C──────────┤
 3: ──RX(3)─────RY(0)─────────────────╰X──╭C──────┤
 4: ──RX(3)─────RY(0)─────────────────────╰X──╭C──┤
 5: ──RX(3)─────RY(0)─────────────────────────╰X──┤
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:241: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
tensor(1., requires_grad=True)
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:241: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Current cost: 0.9999999999972213.
Initial cost: 0.9999999999999843.
Difference: 2.763012041384627e-12
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:366: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
0.9957
 0: ──RX(0.44)──RY(-0.00321)──╭C──────────────────╭┤ Probs
 1: ──RX(3.01)──RY(-4e-05)────╰X──╭C──────────────╰┤ Probs
 2: ──RX(3)─────RY(0)─────────────╰X──╭C───────────┤
 3: ──RX(3)─────RY(0)─────────────────╰X──╭C───────┤
 4: ──RX(3)─────RY(0)─────────────────────╰X──╭C───┤
 5: ──RX(3)─────RY(0)─────────────────────────╰X───┤
Cost after step    10:  0.9909000. Locality: 2
Cost after step    20:  0.9753000. Locality: 2
Cost after step    30:  0.9275000. Locality: 2
Cost after step    40:  0.8386000. Locality: 2
Cost after step    50:  0.6821000. Locality: 2
Cost after step    60:  0.4353000. Locality: 2
Cost after step    70:  0.2264000. Locality: 2
Cost after step    80:  0.0923000. Locality: 2
---Switching Locality---
Cost after step    90:  0.9901000. Locality: 3
Cost after step   100:  0.9737000. Locality: 3
Cost after step   110:  0.9400000. Locality: 3
Cost after step   120:  0.8711000. Locality: 3
Cost after step   130:  0.7228000. Locality: 3
Cost after step   140:  0.5156000. Locality: 3
Cost after step   150:  0.2846000. Locality: 3
Cost after step   160:  0.1285000. Locality: 3
---Switching Locality---
Cost after step   170:  0.9899000. Locality: 4
Cost after step   180:  0.9799000. Locality: 4
Cost after step   190:  0.9512000. Locality: 4
Cost after step   200:  0.8964000. Locality: 4
Cost after step   210:  0.7683000. Locality: 4
Cost after step   220:  0.5752000. Locality: 4
Cost after step   230:  0.3314000. Locality: 4
Cost after step   240:  0.1575000. Locality: 4
---Switching Locality---
Cost after step   250:  0.9942000. Locality: 5
Cost after step   260:  0.9866000. Locality: 5
Cost after step   270:  0.9641000. Locality: 5
Cost after step   280:  0.9120000. Locality: 5
Cost after step   290:  0.8136000. Locality: 5
Cost after step   300:  0.6380000. Locality: 5
Cost after step   310:  0.4004000. Locality: 5
Cost after step   320:  0.1996000. Locality: 5
---Switching Locality---
Cost after step   330:  0.9945000. Locality: 6
Cost after step   340:  0.9873000. Locality: 6
Cost after step   350:  0.9689000. Locality: 6
Cost after step   360:  0.9288000. Locality: 6
Cost after step   370:  0.8476000. Locality: 6
Cost after step   380:  0.6711000. Locality: 6
Cost after step   390:  0.4527000. Locality: 6
Cost after step   400:  0.2342000. Locality: 6
Cost after step   410:  0.1014000. Locality: 6
 0: ──RX(0.00069)──RY(-0.00297)──╭C──────────────────╭┤ Probs
 1: ──RX(0.00331)──RY(-0.0017)───╰X──╭C──────────────├┤ Probs
 2: ──RX(0.017)────RY(-0.00032)──────╰X──╭C──────────├┤ Probs
 3: ──RX(0.0496)───RY(4.5e-05)───────────╰X──╭C──────├┤ Probs
 4: ──RX(0.174)────RY(0.00258)───────────────╰X──╭C──├┤ Probs
 5: ──RX(0.599)────RY(0.0005)────────────────────╰X──╰┤ Probs
--- New run! ---
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/_grad.py:96: UserWarning: Starting with PennyLane v0.20.0, when using Autograd, inputs have to explicitly specify requires_grad=True (or the argnum argument must be passed) in order for trainable parameters to be identified.
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:241: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
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
/home/runner/work/qml/qml/demonstrations/tutorial_local_cost_functions.py:366: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
---Switching Locality---
Cost after step    10:  0.6137000. Locality: 2
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
Cost after step    80:  0.5100000. Locality: 6
---Switching Locality---
Cost after step    90:  0.4677000. Locality: 8
Cost after step   100:  0.1562000. Locality: 8
Trained:     2
Plateau'd:     0
--- New run! ---
---Switching Locality---
---Switching Locality---
---Switching Locality---
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
Cost after step    70:  0.6102000. Locality: 4
---Switching Locality---
Cost after step    80:  0.4909000. Locality: 5
---Switching Locality---
Cost after step    90:  0.7897000. Locality: 6
Cost after step   100:  0.4939000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7063000. Locality: 7
Cost after step   120:  0.5323000. Locality: 7
Cost after step   130:  0.2921000. Locality: 8
Trained:     5
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.7449000. Locality: 2
Cost after step    20:  0.5005000. Locality: 2
---Switching Locality---
---Switching Locality---
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
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
Cost after step    10:  0.6986000. Locality: 2
Cost after step    20:  0.6273000. Locality: 2
Cost after step    30:  0.5645000. Locality: 2
Cost after step    40:  0.5016000. Locality: 2
Cost after step    50:  0.9048000. Locality: 3
Cost after step    60:  0.7903000. Locality: 3
Cost after step    70:  0.5948000. Locality: 3
---Switching Locality---
---Switching Locality---
Cost after step    80:  0.5600000. Locality: 5
---Switching Locality---
Cost after step    90:  0.8307000. Locality: 6
Cost after step   100:  0.6442000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7075000. Locality: 7
Cost after step   120:  0.5936000. Locality: 7
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
Cost after step    10:  0.6759000. Locality: 2
Cost after step    20:  0.8020000. Locality: 3
Cost after step    30:  0.6412000. Locality: 3
Cost after step    40:  0.6920000. Locality: 4
---Switching Locality---
Cost after step    50:  0.7176000. Locality: 5
Cost after step    60:  0.5973000. Locality: 5
---Switching Locality---
Cost after step    70:  0.5469000. Locality: 6
 </code>
 </pre>
 </details>

---

## 28. tutorial_quantum_metrology.html <a name="demo27"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_metrology.html):

```
0: ──H──RZ(0)──H──RX(1.57)──RZ(1)──RX(-1.57)──H───────────────────────────────────────────────────────────╭X──RZ(8)──╭X──H──H─────────╭X──RZ(9)──╭X──H──────────H──╭X──RZ(10)──╭X──H──H─────────╭X──RZ(11)──╭X──H──────────H──────╭X──RZ(12)──╭X───H──H────────────────╭X──RZ(13)──╭X───H──RZ(0)──────PhaseDamp(0.2)──H───────────────RZ(14)──H───────RX(1.57)──RZ(15)────RX(-1.57)─────────────╭┤ Probs
1: ──H──RZ(2)──H──RX(1.57)──RZ(3)──RX(-1.57)──H──╭X──RZ(6)──╭X──H──H─────────╭X──RZ(7)──╭X──H──────────H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H──│───────────│────────────────│───────────│─────────────────╭X──╰C──────────╰C──╭X──H──H─────────╭X──╰C──────────╰C──╭X──H──────────RZ(0)───────────PhaseDamp(0.2)──H───────RZ(16)──H─────────RX(1.57)──RZ(17)─────RX(-1.57)──├┤ Probs
2: ──H──RZ(4)──H──RX(1.57)──RZ(5)──RX(-1.57)──H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H───────────────────────────────────────────────────────────╰C──────────╰C──H──RX(1.57)──╰C──────────╰C──RX(-1.57)──H──╰C──────────────────╰C──H──RX(1.57)──╰C──────────────────╰C──RX(-1.57)──RZ(0)───────────PhaseDamp(0.2)──H───────RZ(18)──H─────────RX(1.57)──RZ(19)─────RX(-1.57)──╰┤ Probs
Initialization: Cost = 3.9901
Iteration    5: Cost = 1.8267
Iteration   10: Cost = 1.7671
Iteration   15: Cost = 1.7988
Iteration   20: Cost = 1.6231
Cost for standard Ramsey sensing = 1.5543
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_metrology.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_quantum_metrology.py:180: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_quantum_metrology.py:181: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_quantum_metrology.py:182: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/operation.py:730: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
 0: ──H──RZ(0)──H──RX(1.57)──RZ(1)──RX(-1.57)──H───────────────────────────────────────────────────────────╭X──RZ(8)──╭X──H──H─────────╭X──RZ(9)──╭X──H──────────H──╭X──RZ(10)──╭X──H──H─────────╭X──RZ(11)──╭X──H──────────H──────╭X──RZ(12)──╭X───H──H────────────────╭X──RZ(13)──╭X───H──RZ(0)──────PhaseDamp(0.2)──H───────────────RZ(14)──H───────RX(1.57)──RZ(15)────RX(-1.57)─────────────╭┤ Probs
 1: ──H──RZ(2)──H──RX(1.57)──RZ(3)──RX(-1.57)──H──╭X──RZ(6)──╭X──H──H─────────╭X──RZ(7)──╭X──H──────────H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H──│───────────│────────────────│───────────│─────────────────╭X──╰C──────────╰C──╭X──H──H─────────╭X──╰C──────────╰C──╭X──H──────────RZ(0)───────────PhaseDamp(0.2)──H───────RZ(16)──H─────────RX(1.57)──RZ(17)─────RX(-1.57)──├┤ Probs
 2: ──H──RZ(4)──H──RX(1.57)──RZ(5)──RX(-1.57)──H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H───────────────────────────────────────────────────────────╰C──────────╰C──H──RX(1.57)──╰C──────────╰C──RX(-1.57)──H──╰C──────────────────╰C──H──RX(1.57)──╰C──────────────────╰C──RX(-1.57)──RZ(0)───────────PhaseDamp(0.2)──H───────RZ(18)──H─────────RX(1.57)──RZ(19)─────RX(-1.57)──╰┤ Probs
/home/runner/work/qml/qml/demonstrations/tutorial_quantum_metrology.py:180: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_quantum_metrology.py:181: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
```

---

## 29. tutorial_rosalin.html <a name="demo28"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
-0.8395887630997874
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
/home/runner/work/qml/qml/demonstrations/tutorial_rosalin.py:217: UserWarning: The init module will be deprecated soon, since templates can now provide a method that returns the shape of parameter tensors.
-0.8395887630997874
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
Step 97: cost = -7.885310745063636 shots_used = 235200
Step 98: cost = -7.883507674089134 shots_used = 237600
 </code>
 </pre>
 </details>

---

## 30. tutorial_quantum_chemistry.html <a name="demo29"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
(-46.463906788688945+0j) [] +
(-0.01458364890761256+0j) [X0 X1 Y2 Y3] +
(-3.570761328913366e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.00565262097801732+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209823+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939577780457e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761328913366e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017319+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209823+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577780457e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186814+0j) [X0 X1 Y4 Y5] +
(-2.4473231286791434e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104565368e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.00380406617172854+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231286791434e-07+0j) [X0 X1 X5 X6] +
(-7.867765104565368e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.00380406617172854+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970575+0j) [X0 X1 Y6 Y7] +
(-7.735036880590954e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.703578355309803e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880590954e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783553098026e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0065093612011772346+0j) [X0 X1 Y8 Y9] +
(-0.0077314252507752765+0j) [X0 X1 Y10 Y11] +
(5.627851911600192e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911600192e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402953+0j) [X0 X1 Y12 Y13] +
(0.01458364890761256+0j) [X0 Y1 Y2 X3] +
(3.570761328913366e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.00565262097801732+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209823+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939577780457e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761328913366e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017319+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209823+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577780457e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186814+0j) [X0 Y1 Y4 X5] +
(2.4473231286791434e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104565368e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.00380406617172854+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231286791434e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104565368e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.00380406617172854+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970575+0j) [X0 Y1 Y6 X7] +
(7.735036880590954e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.703578355309803e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880590954e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783553098026e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0065093612011772346+0j) [X0 Y1 Y8 X9] +
(0.0077314252507752765+0j) [X0 Y1 Y10 X11] +
(-5.627851911600192e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911600192e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402953+0j) [X0 Y1 Y12 X13] +
(0.1250703257977161+0j) [X0 Z1 X2] +
(-1.9332412769736937e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.002293956611352449+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.001640754855312387+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714591024744e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412769736937e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.002293956611352449+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.001640754855312387+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714591024744e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315555+0j) [X0 Z1 X2 Z3] +
(-1.5510539175768164e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376508462162e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770577+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480004474e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128987086174e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676587+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631474+0j) [X0 Z1 X2 Z4] +
(-1.3807781480004474e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.37673930865907e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458707+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480004474e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.37673930865907e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458707+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896186+0j) [X0 Z1 X2 Z5] +
(0.005708495985960883+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332103810461e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253797259382e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076801+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305986591849e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00803252091882132+0j) [X0 Z1 X2 Z6] +
(0.000594022154300514+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773244928102e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.000594022154300514+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244928102e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661033+0j) [X0 Z1 X2 Z7] +
(0.011055020596131983+0j) [X0 Z1 X2 Z8] +
(0.0029297686747509627+0j) [X0 Z1 X2 Z9] +
(-6.418291574866951e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914889864e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504174+0j) [X0 Z1 X2 Z10] +
(-1.1076325599515788e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599515788e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018411668+0j) [X0 Z1 X2 Z11] +
(0.0069012382497971965+0j) [X0 Z1 X2 Z12] +
(0.0023262306231580055+0j) [X0 Z1 X2 Z13] +
(-3.568247521375991e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939904+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716556096567e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408163+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253796601476e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441856+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.5233896784271043e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217879+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199803095e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118695+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155216+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776287+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975893227e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660369+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692465523466e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.00812525192138102+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630075+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744939785e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624913874e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639191+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441856+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.5233896784271043e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217879+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199803095e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118695+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155216+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776287+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975893227e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660369+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692465523466e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.00812525192138102+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630075+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744939785e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624913874e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639191+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768791710645e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125537+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.000787089677102452+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125537+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.000787089677102452+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694863771494e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978541626377e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441798+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.684915095390719e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004515+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153864515e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616333555e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.00239497263979802+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616333555e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.00239497263979802+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961069484e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310132540348e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.001303800478812697+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619308+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197743199434e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823568+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823568+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453083555383e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217261212e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536652278777e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562715+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.8394209153864512e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029756838+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538398+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478154493e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446596440516e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369514+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696595+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265649824355e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209153864512e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029756838+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538398+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289478154493e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446596440516e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369514+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696595+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265649824355e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.042743277013782895+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487864+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.850564192885799e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487864+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192885799e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255696+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182545+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(2.312094305338397e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282185578141e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839347+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.24697442595074e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.24697442595074e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803869+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914293+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907482+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.200428749440663e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328875+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303547829+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207392926e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422587026e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423545+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328875+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303547829+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207392926e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422587026e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423545+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336959+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341414158406e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336959+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341414158406e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002465+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231016106+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046447+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245081+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.002984166168121939+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.002984166168121939+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009013869307e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476488293675e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.87662165826669e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213184739e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.001532483523073012+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998839886594e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.0054089544224099695+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298147067e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278084+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037243739e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226871+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079230129538e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213667+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221157824e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317542396595e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339204+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.000715673424890874+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325315961033e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718675841076e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496512+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389547468+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309319011966e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332621995272e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440638+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169124+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.6704023910036716e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651348+0j) [X0 X2] +
(3.117447945874658e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129795+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.05859198873386186+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453502218e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01458364890761256+0j) [Y0 X1 X2 Y3] +
(3.570761328913366e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.00565262097801732+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209823+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939577780457e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761328913366e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017319+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209823+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577780457e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186814+0j) [Y0 X1 X4 Y5] +
(2.4473231286791434e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104565368e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.00380406617172854+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231286791434e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104565368e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.00380406617172854+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970575+0j) [Y0 X1 X6 Y7] +
(7.735036880590954e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.703578355309803e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880590954e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783553098026e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0065093612011772346+0j) [Y0 X1 X8 Y9] +
(0.0077314252507752765+0j) [Y0 X1 X10 Y11] +
(-5.627851911600192e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911600192e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402953+0j) [Y0 X1 X12 Y13] +
(-0.01458364890761256+0j) [Y0 Y1 X2 X3] +
(-3.570761328913366e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.00565262097801732+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209823+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939577780457e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761328913366e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017319+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209823+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577780457e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186814+0j) [Y0 Y1 X4 X5] +
(-2.4473231286791434e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104565368e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.00380406617172854+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231286791434e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104565368e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.00380406617172854+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970575+0j) [Y0 Y1 X6 X7] +
(-7.735036880590954e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.703578355309803e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880590954e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783553098026e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0065093612011772346+0j) [Y0 Y1 X8 X9] +
(-0.0077314252507752765+0j) [Y0 Y1 X10 X11] +
(5.627851911600192e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911600192e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402953+0j) [Y0 Y1 X12 X13] +
(-3.568247521375991e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939904+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408163+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253796601476e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716556096567e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1250703257977161+0j) [Y0 Z1 Y2] +
(-1.9332412769736937e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.002293956611352449+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.001640754855312387+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714591024744e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412769736937e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.002293956611352449+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.001640754855312387+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714591024744e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315555+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781480004474e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128987086174e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676587+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539175768164e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376508462162e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770577+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631474+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480004474e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.37673930865907e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458707+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480004474e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.37673930865907e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458707+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896186+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076801+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305986591849e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960883+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253797259382e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332103810461e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00803252091882132+0j) [Y0 Z1 Y2 Z6] +
(0.000594022154300514+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773244928102e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.000594022154300514+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244928102e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661033+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596131983+0j) [Y0 Z1 Y2 Z8] +
(0.0029297686747509627+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914889864e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574866951e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504174+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599515788e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599515788e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018411668+0j) [Y0 Z1 Y2 Z11] +
(0.0069012382497971965+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231580055+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441856+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.5233896784271043e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217879+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199803095e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118695+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155216+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776287+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975893227e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660369+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692465523466e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.00812525192138102+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630075+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744939785e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624913874e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639191+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441856+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.5233896784271043e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217879+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199803095e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118695+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155216+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776287+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975893227e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660369+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692465523466e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.00812525192138102+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630075+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744939785e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624913874e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639191+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562715+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.2020768791710645e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125537+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.000787089677102452+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125537+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.000787089677102452+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694863771494e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.684915095390719e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004515+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978541626377e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441798+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153864515e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616333555e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.00239497263979802+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616333555e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.00239497263979802+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961069484e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310132540348e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619308+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.001303800478812697+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197743199434e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823568+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823568+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453083555383e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217261212e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536652278777e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209153864512e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029756838+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538398+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289478154493e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446596440516e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369514+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696595+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265649824355e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209153864512e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029756838+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538398+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478154493e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446596440516e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369514+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696595+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265649824355e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.200428749440663e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.042743277013782895+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487864+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.850564192885799e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487864+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192885799e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255696+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182545+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(1.0717282185578141e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.312094305338397e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839347+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.24697442595074e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.24697442595074e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803869+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914293+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907482+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328875+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303547829+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207392926e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422587026e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423545+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328875+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303547829+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207392926e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422587026e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423545+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336959+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341414158406e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336959+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341414158406e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002465+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231016106+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046447+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245081+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.002984166168121939+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.002984166168121939+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009013869307e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476488293675e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.87662165826669e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213184739e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.001532483523073012+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998839886594e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.0054089544224099695+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298147067e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278084+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037243739e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226871+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079230129538e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213667+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221157824e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317542396595e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339204+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.000715673424890874+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325315961033e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718675841076e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496512+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389547468+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309319011966e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332621995272e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440638+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169124+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.6704023910036716e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651348+0j) [Y0 Y2] +
(3.117447945874658e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129795+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.05859198873386186+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453502218e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111766+0j) [Z0] +
(0.10433064780651348+0j) [Z0 X1 Z2 X3] +
(3.117447945874658e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.045879470781297955+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.05859198873386186+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453502218e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651348+0j) [Z0 Y1 Z2 Y3] +
(3.117447945874658e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.045879470781297955+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.05859198873386186+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453502218e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860487+0j) [Z0 Z1] +
(-8.337746754806007e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273006+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214034+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735878409e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746754806007e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273006+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214034+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735878409e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830376+0j) [Z0 Z2] +
(-1.1908508083719372e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329032+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635015+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603693656453e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508083719372e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329032+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635015+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603693656453e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.25129445674591633+0j) [Z0 Z3] +
(-3.0993492435174123e-06+0j) [Z0 X4 Z5 X6] +
(-1.53168087964101e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863624+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.0993492435174123e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.53168087964101e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863624+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342148+0j) [Z0 Z4] +
(-3.3440815563853266e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585306866635e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036477+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815563853266e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585306866635e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036477+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1993635453736083+0j) [Z0 Z5] +
(0.056084681246613366+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209670252007e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613366+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209670252007e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2416466393601721+0j) [Z0 Z6] +
(0.05600733087780746+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851834721026e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780746+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851834721026e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314267+0j) [Z0 Z7] +
(0.2723251830660567+0j) [Z0 Z8] +
(0.278834544267234+0j) [Z0 Z9] +
(-2.1776646052655865e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646052655865e-06+0j) [Z0 Y10 Z11 Y12] +
(0.1929972393536422+0j) [Z0 Z10] +
(-1.614879414105567e-06+0j) [Z0 X11 Z12 X13] +
(-1.614879414105567e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441752+0j) [Z0 Z11] +
(0.21102659849791497+0j) [Z0 Z12] +
(0.21631037498631792+0j) [Z0 Z13] +
(1.933241276973694e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352449+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553123872+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714591024744e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441856+0j) [X1 X2 X4 X5] +
(-8.091637199803095e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118695+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.5233896784271043e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217879+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155216+0j) [X1 X2 X6 X7] +
(0.005114473831660369+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465523466e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776287+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975893227e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.00812525192138102+0j) [X1 X2 X8 X9] +
(-0.0017992194936630073+0j) [X1 X2 X10 X11] +
(-5.287660624913874e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744939785e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639191+0j) [X1 X2 X12 X13] +
(-1.933241276973694e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352449+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553123872+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714591024744e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441856+0j) [X1 Y2 Y4 X5] +
(-8.091637199803095e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118695+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5233896784271043e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217879+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155216+0j) [X1 Y2 Y6 X7] +
(0.005114473831660369+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465523466e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776287+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975893227e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.00812525192138102+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630073+0j) [X1 Y2 Y10 X11] +
(-5.287660624913874e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744939785e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639191+0j) [X1 Y2 Y12 X13] +
(0.12507032579771618+0j) [X1 Z2 X3] +
(-1.3807781480004474e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.37673930865907e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458707+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480004474e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.37673930865907e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458707+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896186+0j) [X1 Z2 X3 Z4] +
(-1.5510539175768164e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376508462162e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770577+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480004474e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128987086174e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676587+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631474+0j) [X1 Z2 X3 Z5] +
(0.000594022154300514+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773244928102e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.000594022154300514+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244928102e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661033+0j) [X1 Z2 X3 Z6] +
(0.005708495985960883+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332103810461e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253797259382e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076801+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305986591849e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00803252091882132+0j) [X1 Z2 X3 Z7] +
(0.0029297686747509627+0j) [X1 Z2 X3 Z8] +
(0.011055020596131983+0j) [X1 Z2 X3 Z9] +
(-1.1076325599515788e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599515788e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018411668+0j) [X1 Z2 X3 Z10] +
(-6.418291574866951e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914889864e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504174+0j) [X1 Z2 X3 Z11] +
(0.0023262306231580055+0j) [X1 Z2 X3 Z12] +
(0.0069012382497971965+0j) [X1 Z2 X3 Z13] +
(-3.568247521375991e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939904+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716556096567e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408163+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253796601476e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125537+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.000787089677102452+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209153864517e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538398+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029756838+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478154493e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446596440517e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696595+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369514+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826564982435e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125537+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.000787089677102452+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209153864517e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538398+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029756838+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478154493e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446596440517e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696595+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369514+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826564982435e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076879171065e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616333555e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.00239497263979802+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616333555e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.00239497263979802+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978541626377e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441798+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.684915095390719e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004515+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153864515e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310132540348e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961069484e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823568+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823568+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453083555383e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.001303800478812697+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619308+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197743199434e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536652278777e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217261212e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562715+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0005192743499487864+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.850564192885799e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832888+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303547829+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422587026e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207392926e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423545+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487864+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.850564192885799e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832888+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303547829+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422587026e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207392926e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423545+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378291+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.004636976661182545+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.24697442595074e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.24697442595074e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803869+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.312094305338397e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282185578141e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839347+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907482+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914293+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.200428749440663e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336959+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341414158406e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336959+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341414158406e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.002984166168121939+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.002984166168121939+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002471+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245081+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046447+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009013869307e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476488293675e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213184739e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231016106+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.87662165826669e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.0054089544224099695+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298147067e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.001532483523073012+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998839886594e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226871+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079230129538e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025569+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278084+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037243739e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339204+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.000715673424890874+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325315961033e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694863771497e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213667+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221157824e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317542396595e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332621995272e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440638+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169124+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.6704023910036716e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.003276971931231555+0j) [X1 X3] +
(3.6060718675841076e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496512+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389547468+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309319011966e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241276973694e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352449+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553123872+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714591024744e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441856+0j) [Y1 X2 X4 Y5] +
(-8.091637199803095e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118695+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.5233896784271043e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217879+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155216+0j) [Y1 X2 X6 Y7] +
(0.005114473831660369+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465523466e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776287+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975893227e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.00812525192138102+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630073+0j) [Y1 X2 X10 Y11] +
(-5.287660624913874e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744939785e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639191+0j) [Y1 X2 X12 Y13] +
(1.933241276973694e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352449+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553123872+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714591024744e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441856+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199803095e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118695+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.5233896784271043e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217879+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155216+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660369+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465523466e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776287+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975893227e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.00812525192138102+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630073+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624913874e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744939785e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639191+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521375991e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939904+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408163+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253796601476e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716556096567e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579771618+0j) [Y1 Z2 Y3] +
(-1.3807781480004474e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.37673930865907e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458707+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480004474e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.37673930865907e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458707+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896186+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781480004474e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128987086174e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676587+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539175768164e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376508462162e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770577+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631474+0j) [Y1 Z2 Y3 Z5] +
(0.000594022154300514+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773244928102e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.000594022154300514+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244928102e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661033+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076801+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305986591849e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960883+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253797259382e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332103810461e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00803252091882132+0j) [Y1 Z2 Y3 Z7] +
(0.0029297686747509627+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596131983+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599515788e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599515788e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018411668+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914889864e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574866951e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504174+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231580055+0j) [Y1 Z2 Y3 Z12] +
(0.0069012382497971965+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125537+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.000787089677102452+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209153864517e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538398+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029756838+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478154493e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446596440517e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696595+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369514+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826564982435e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125537+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.000787089677102452+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209153864517e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538398+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029756838+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478154493e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446596440517e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696595+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369514+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826564982435e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562715+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.202076879171065e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616333555e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.00239497263979802+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616333555e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.00239497263979802+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.684915095390719e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004515+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978541626377e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441798+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153864515e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310132540348e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961069484e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823568+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823568+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453083555383e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619308+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.001303800478812697+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197743199434e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536652278777e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217261212e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487864+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.850564192885799e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832888+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303547829+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422587026e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207392926e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423545+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487864+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.850564192885799e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832888+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303547829+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422587026e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207392926e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423545+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.200428749440663e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378291+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.004636976661182545+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.24697442595074e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.24697442595074e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803869+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282185578141e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.312094305338397e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839347+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907482+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914293+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336959+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341414158406e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336959+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341414158406e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.002984166168121939+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.002984166168121939+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002471+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245081+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046447+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009013869307e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476488293675e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213184739e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231016106+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.87662165826669e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.0054089544224099695+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298147067e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.001532483523073012+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998839886594e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226871+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079230129538e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025569+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278084+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037243739e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339204+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.000715673424890874+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325315961033e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694863771497e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213667+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221157824e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317542396595e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332621995272e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440638+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169124+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.6704023910036716e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231555+0j) [Y1 Y3] +
(3.6060718675841076e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496512+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389547468+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309319011966e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111766+0j) [Z1] +
(-1.1908508083719372e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329032+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635015+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603693656453e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508083719372e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329032+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635015+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603693656453e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.25129445674591633+0j) [Z1 Z2] +
(-8.337746754806007e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273006+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214034+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735878409e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746754806007e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273006+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214034+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735878409e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830376+0j) [Z1 Z3] +
(-3.3440815563853266e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585306866635e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036477+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815563853266e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585306866635e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036477+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1993635453736083+0j) [Z1 Z4] +
(-3.0993492435174123e-06+0j) [Z1 X5 Z6 X7] +
(-1.53168087964101e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863624+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.0993492435174123e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.53168087964101e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863624+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342148+0j) [Z1 Z5] +
(0.05600733087780746+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851834721026e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780746+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851834721026e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314267+0j) [Z1 Z6] +
(0.056084681246613366+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209670252007e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613366+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209670252007e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2416466393601721+0j) [Z1 Z7] +
(0.278834544267234+0j) [Z1 Z8] +
(0.2723251830660567+0j) [Z1 Z9] +
(-1.614879414105567e-06+0j) [Z1 X10 Z11 X12] +
(-1.614879414105567e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441752+0j) [Z1 Z10] +
(-2.1776646052655865e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646052655865e-06+0j) [Z1 Y11 Z12 Y13] +
(0.1929972393536422+0j) [Z1 Z11] +
(0.21631037498631792+0j) [Z1 Z12] +
(0.21102659849791497+0j) [Z1 Z13] +
(-0.03583956795335346+0j) [X2 X3 Y4 Y5] +
(-2.1990516189242654e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563204800494e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01031148248983172+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516189242654e-07+0j) [X2 X3 X5 X6] +
(-2.3609563204800494e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.01031148248983172+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.03114381798896706+0j) [X2 X3 Y6 Y7] +
(0.005368659358109472+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350635313922e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109472+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350635313922e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904255+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457413+0j) [X2 X3 Y10 Y11] +
(2.172669101602139e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.172669101602139e-06+0j) [X2 X3 X11 X12] +
(-0.01557720806397643+0j) [X2 X3 Y12 Y13] +
(0.03583956795335346+0j) [X2 Y3 Y4 X5] +
(2.1990516189242654e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563204800494e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.01031148248983172+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516189242654e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563204800494e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.01031148248983172+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03114381798896706+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109472+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350635313922e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109472+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350635313922e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904255+0j) [X2 Y3 Y8 X9] +
(0.025384657508457413+0j) [X2 Y3 Y10 X11] +
(-2.172669101602139e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.172669101602139e-06+0j) [X2 Y3 Y11 X12] +
(0.01557720806397643+0j) [X2 Y3 Y12 X13] +
(-3.887051673410292e-06+0j) [X2 Z3 X4] +
(-0.005143391768825049+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962602+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706340191e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825049+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962602+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706340191e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411980009e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489515893967e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.01075756395390889+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178094780995e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411217487e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534390455372e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420193039188e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363717+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420193039188e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363717+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489009937739e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423769754266e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052995309166e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380213+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.0053248352342216716+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564323230993e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678069032+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678069032+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707501142997e-06+0j) [X2 Z3 X4 Z11] +
(3.539054184694308e-06+0j) [X2 Z3 X4 Z12] +
(8.814937307126514e-06+0j) [X2 Z3 X4 Z13] +
(1.628853243658932e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796714+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01026341486815854+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424489832764e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346311373454e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251544+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930677145229e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454832+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372284593e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068819898e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847355+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688815+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122432207e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424489832764e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346311373454e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251544+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930677145229e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454832+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372284593e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068819898e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847355+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688815+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122432207e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042326+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023786+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815471852463e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023786+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815471852463e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802121+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826853+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646086+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288287389e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323108836091e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956243+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184006938962e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184006938962e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.01441109943013098+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499334+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.003493790359890098+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447179451299e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819232+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226605+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507115775055e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429522635e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0041587973818400315+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819232+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226605+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507115775055e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429522635e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0041587973818400315+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162148+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990715532572e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162148+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990715532572e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702303+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946563423355e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946563423355e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.0242821173546929+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.01953805031131471+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898817+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.002446497155415895+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.002446497155415895+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.775950527537525e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765761652415e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327810055e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.84620167146772e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205312+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793767771e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.024755463292890974+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526722214514e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721601027+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350502073987e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.02990378951262481+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656792512e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907702+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942905+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560119086636e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334279+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905505+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718248854e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167405307327e-06+0j) [X2 X4] +
(0.0004956762314917413+0j) [X2 Z4 Z5 X6] +
(-0.0356083789883125+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.2532733484169045e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335346+0j) [Y2 X3 X4 Y5] +
(2.1990516189242654e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563204800494e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.01031148248983172+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516189242654e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563204800494e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.01031148248983172+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.03114381798896706+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109472+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350635313922e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109472+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350635313922e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904255+0j) [Y2 X3 X8 Y9] +
(0.025384657508457413+0j) [Y2 X3 X10 Y11] +
(-2.172669101602139e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.172669101602139e-06+0j) [Y2 X3 X11 Y12] +
(0.01557720806397643+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335346+0j) [Y2 Y3 X4 X5] +
(-2.1990516189242654e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563204800494e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01031148248983172+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516189242654e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563204800494e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.01031148248983172+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.03114381798896706+0j) [Y2 Y3 X6 X7] +
(0.005368659358109472+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350635313922e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109472+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350635313922e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904255+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457413+0j) [Y2 Y3 X10 X11] +
(2.172669101602139e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.172669101602139e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.01557720806397643+0j) [Y2 Y3 X12 X13] +
(1.628853243658932e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796714+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.01026341486815854+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051673410292e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825049+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962602+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706340191e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825049+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962602+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706340191e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411980009e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178094780995e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411217487e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489515893967e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.01075756395390889+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534390455372e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420193039188e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363717+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420193039188e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363717+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489009937739e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423769754266e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052995309166e-07+0j) [Y2 Z3 Y4 Z9] +
(0.0053248352342216716+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380213+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564323230993e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678069032+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678069032+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707501142997e-06+0j) [Y2 Z3 Y4 Z11] +
(3.539054184694308e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937307126514e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424489832764e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346311373454e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251544+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930677145229e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454832+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372284593e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068819898e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847355+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688815+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122432207e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424489832764e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346311373454e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251544+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930677145229e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454832+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372284593e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068819898e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847355+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688815+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122432207e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447179451299e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042326+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023786+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815471852463e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023786+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815471852463e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802121+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826853+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646086+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323108836091e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288287389e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956243+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184006938962e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184006938962e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.01441109943013098+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499334+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.003493790359890098+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819232+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226605+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507115775055e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429522635e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0041587973818400315+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819232+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226605+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507115775055e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429522635e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0041587973818400315+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162148+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990715532572e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162148+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990715532572e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702303+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946563423355e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946563423355e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.0242821173546929+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.01953805031131471+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898817+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.002446497155415895+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.002446497155415895+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.775950527537525e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765761652415e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327810055e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.84620167146772e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205312+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793767771e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.024755463292890974+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526722214514e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721601027+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350502073987e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.02990378951262481+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656792512e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907702+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942905+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560119086636e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334279+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905505+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718248854e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167405307327e-06+0j) [Y2 Y4] +
(0.0004956762314917413+0j) [Y2 Z4 Z5 Y6] +
(-0.0356083789883125+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.2532733484169045e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831703+0j) [Z2] +
(1.6021167405307327e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314917413+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.0356083789883125+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348416904e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167405307327e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314917413+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.0356083789883125+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348416904e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751316+0j) [Z2 Z3] +
(-9.509249751272195e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843147492999e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.02459186088382997+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751272195e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147492999e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.02459186088382997+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503198+0j) [Z2 Z4] +
(-1.170830137019646e-06+0j) [Z2 X5 Z6 X7] +
(-7.0897994679730504e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.034903343373661695+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.170830137019646e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.0897994679730504e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.034903343373661695+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838545+0j) [Z2 Z5] +
(0.019020423173039886+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.103215604947086e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039886+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.103215604947086e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683202+0j) [Z2 Z6] +
(0.02438908253114936+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220985939467e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.02438908253114936+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220985939467e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579908+0j) [Z2 Z7] +
(0.15071408121008265+0j) [Z2 Z8] +
(0.18690820476912523+0j) [Z2 Z9] +
(-1.0632283425299046e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283425299046e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468395+0j) [Z2 Z10] +
(1.1094407590722344e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407590722344e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314137+0j) [Z2 Z11] +
(0.14011289865354787+0j) [Z2 Z12] +
(0.1556901067175243+0j) [Z2 Z13] +
(0.005143391768825049+0j) [X3 X4 Y5 Y6] +
(0.009841749246962602+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706340191e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842448983276e-06+0j) [X3 X4 X6 X7] +
(-1.5224930677145229e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454832+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346311373454e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251544+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372284593e-07+0j) [X3 X4 X8 X9] +
(-4.643051068819898e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688815+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847355+0j) [X3 X4 Y11 Y12] +
(5.275883122432206e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825049+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962602+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706340191e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842448983276e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930677145229e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454832+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346311373454e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251544+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372284593e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068819898e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688815+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847355+0j) [X3 Y4 Y11 X12] +
(5.275883122432206e-06+0j) [X3 Y4 Y12 X13] +
(-3.8870516734102925e-06+0j) [X3 Z4 X5] +
(3.2118420193039188e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363717+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420193039188e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363717+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489009937739e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489515893967e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.01075756395390889+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178094780995e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411217487e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534390455372e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052995309166e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423769754266e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678069032+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678069032+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707501142997e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380213+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.0053248352342216716+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564323230993e-06+0j) [X3 Z4 X5 Z11] +
(8.814937307126514e-06+0j) [X3 Z4 X5 Z12] +
(3.539054184694308e-06+0j) [X3 Z4 X5 Z13] +
(1.628853243658932e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796714+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01026341486815854+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023786+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815471852463e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819232+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226605+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429522635e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507115775055e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.0041587973818400315+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023786+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815471852463e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819232+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226605+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429522635e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507115775055e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.0041587973818400315+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042329+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646086+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826853+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184006938962e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184006938962e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.01441109943013098+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288287389e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323108836091e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956243+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.003493790359890098+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499334+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447179451299e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.01460370472916215+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990715532572e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.01460370472916215+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990715532572e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946563423355e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158947+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946563423355e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158947+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702302+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898817+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.01953805031131471+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527537528e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676576165242e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.84620167146772e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354692902+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327810055e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.024755463292890974+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526722214514e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205312+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793767771e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.02990378951262481+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656792512e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021208+0j) [X3 Z4 Z5 X7] +
(-0.021433810721601027+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350502073987e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334279+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905505+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718248854e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.76499411980009e-07+0j) [X3 X5] +
(0.0016638798784907702+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942905+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560119086636e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825049+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962602+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706340191e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842448983276e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930677145229e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454832+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346311373454e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251544+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372284593e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068819898e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688815+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847355+0j) [Y3 X4 X11 Y12] +
(5.275883122432206e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825049+0j) [Y3 Y4 X5 X6] +
(0.009841749246962602+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706340191e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842448983276e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930677145229e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454832+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346311373454e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251544+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372284593e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068819898e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688815+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847355+0j) [Y3 Y4 X11 X12] +
(5.275883122432206e-06+0j) [Y3 Y4 Y12 Y13] +
(1.628853243658932e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796714+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.01026341486815854+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.8870516734102925e-06+0j) [Y3 Z4 Y5] +
(3.2118420193039188e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363717+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420193039188e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363717+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489009937739e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178094780995e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411217487e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489515893967e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.01075756395390889+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534390455372e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052995309166e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423769754266e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678069032+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678069032+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707501142997e-06+0j) [Y3 Z4 Y5 Z10] +
(0.0053248352342216716+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380213+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564323230993e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937307126514e-06+0j) [Y3 Z4 Y5 Z12] +
(3.539054184694308e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023786+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815471852463e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819232+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226605+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429522635e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507115775055e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.0041587973818400315+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023786+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815471852463e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819232+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226605+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429522635e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507115775055e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.0041587973818400315+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447179451299e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042329+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646086+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826853+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184006938962e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184006938962e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.01441109943013098+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323108836091e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288287389e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956243+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.003493790359890098+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499334+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.01460370472916215+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990715532572e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.01460370472916215+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990715532572e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946563423355e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158947+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946563423355e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158947+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702302+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898817+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.01953805031131471+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527537528e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676576165242e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.84620167146772e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354692902+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327810055e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.024755463292890974+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526722214514e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205312+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793767771e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.02990378951262481+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656792512e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021208+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721601027+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350502073987e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334279+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905505+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718248854e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411980009e-07+0j) [Y3 Y5] +
(0.0016638798784907702+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942905+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560119086636e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.653894222683171+0j) [Z3] +
(-1.170830137019646e-06+0j) [Z3 X4 Z5 X6] +
(-7.0897994679730504e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.034903343373661695+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.170830137019646e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.0897994679730504e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.034903343373661695+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838545+0j) [Z3 Z4] +
(-9.509249751272195e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147492999e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02459186088382997+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751272195e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147492999e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02459186088382997+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503198+0j) [Z3 Z5] +
(0.02438908253114936+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220985939467e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.02438908253114936+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220985939467e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579908+0j) [Z3 Z6] +
(0.019020423173039886+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.103215604947086e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039886+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.103215604947086e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683202+0j) [Z3 Z7] +
(0.18690820476912523+0j) [Z3 Z8] +
(0.15071408121008265+0j) [Z3 Z9] +
(1.1094407590722344e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407590722344e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314137+0j) [Z3 Z10] +
(-1.0632283425299046e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283425299046e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468395+0j) [Z3 Z11] +
(0.1556901067175243+0j) [Z3 Z12] +
(0.14011289865354787+0j) [Z3 Z13] +
(-0.011982389010247898+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832921+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935942757554e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832921+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935942757554e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856952+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248153+0j) [X4 X5 Y10 Y11] +
(-3.6945132948180213e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132948180213e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480388+0j) [X4 X5 Y12 Y13] +
(0.011982389010247898+0j) [X4 Y5 Y6 X7] +
(0.007306759928832921+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935942757554e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832921+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935942757554e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856952+0j) [X4 Y5 Y8 X9] +
(0.01768006795248153+0j) [X4 Y5 Y10 X11] +
(3.6945132948180213e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132948180213e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480388+0j) [X4 Y5 Y12 X13] +
(-1.226048498862674e-05+0j) [X4 Z5 X6] +
(-1.2283337824194587e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756958852+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824194587e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756958852+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579269282e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449080895562e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501831631959e-06+0j) [X4 Z5 X6 Z9] +
(0.00796088072592154+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730235+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978286759633e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613865+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613865+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913885102888e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155945533e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694565+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750736396e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713778851e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.01128519020084084+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.0201759217235354+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.55656921851811e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750736396e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713778851e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.01128519020084084+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.0201759217235354+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.55656921851811e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731887369542e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561344+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731887369542e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561344+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928900514e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179573+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179573+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.334331289652168e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622039105318e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775801987e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736696669e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736696669e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156217+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929529027+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847253+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.02563723829602683+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817865196735e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638312+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344676459781e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.041718813839821775+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433759329e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289348+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362216178788e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719759+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765815358775e-07+0j) [X4 X6] +
(-4.253224225796711e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601307+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247898+0j) [Y4 X5 X6 Y7] +
(0.007306759928832921+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935942757554e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832921+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935942757554e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856952+0j) [Y4 X5 X8 Y9] +
(0.01768006795248153+0j) [Y4 X5 X10 Y11] +
(3.6945132948180213e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132948180213e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480388+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247898+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832921+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935942757554e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832921+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935942757554e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856952+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248153+0j) [Y4 Y5 X10 X11] +
(-3.6945132948180213e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132948180213e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480388+0j) [Y4 Y5 X12 X13] +
(0.008890731522694565+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.226048498862674e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824194587e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756958852+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824194587e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756958852+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579269282e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449080895562e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501831631959e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730235+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.00796088072592154+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978286759633e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613865+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613865+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913885102888e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155945533e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750736396e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713778851e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.01128519020084084+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.0201759217235354+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.55656921851811e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750736396e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713778851e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.01128519020084084+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.0201759217235354+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.55656921851811e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731887369542e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561344+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731887369542e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561344+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928900514e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179573+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179573+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.334331289652168e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622039105318e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775801987e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736696669e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736696669e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156217+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929529027+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847253+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.02563723829602683+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817865196735e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638312+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344676459781e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.041718813839821775+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433759329e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289348+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362216178788e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719759+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765815358775e-07+0j) [Y4 Y6] +
(-4.253224225796711e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601307+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145643+0j) [Z4] +
(-5.929765815358775e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225796711e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02252844019601307+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765815358775e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225796711e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02252844019601307+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985667+0j) [Z4 Z5] +
(0.01826683486937551+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174774073516e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01826683486937551+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174774073516e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040755+0j) [Z4 Z6] +
(0.010960074940542594+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468368349273e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542594+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468368349273e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065545+0j) [Z4 Z7] +
(0.15676396176430998+0j) [Z4 Z9] +
(1.8782101248144929e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101248144929e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237605+0j) [Z4 Z10] +
(-1.8163031700035276e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031700035276e-06+0j) [Z4 Y11 Z12 Y13] +
(0.1425799771248576+0j) [Z4 Z11] +
(0.11383573679388652+0j) [Z4 Z12] +
(0.1521504070886904+0j) [Z4 Z13] +
(1.2283337824194587e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756958852+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750736396e-07+0j) [X5 X6 X8 X9] +
(5.974311713778851e-06+0j) [X5 X6 X10 X11] +
(0.0201759217235354+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.01128519020084084+0j) [X5 X6 Y11 Y12] +
(-4.55656921851811e-06+0j) [X5 X6 X12 X13] +
(-1.2283337824194587e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756958852+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750736396e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713778851e-06+0j) [X5 Y6 Y10 X11] +
(0.0201759217235354+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.01128519020084084+0j) [X5 Y6 Y11 X12] +
(-4.55656921851811e-06+0j) [X5 Y6 Y12 X13] +
(-1.226048498862674e-05+0j) [X5 Z6 X7] +
(-1.8818501831631959e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449080895562e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613865+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613865+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913885102888e-06+0j) [X5 Z6 X7 Z10] +
(0.00796088072592154+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730235+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978286759633e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155945533e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694565+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731887369542e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561344+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731887369542e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561344+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179576+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736696669e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179576+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736696669e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928900514e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775801987e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622039105318e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156195+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529027+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.02563723829602683+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312896521674e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847253+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344676459781e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.041718813839821775+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817865196735e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638312+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362216178788e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719759+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608579269282e-06+0j) [X5 X7] +
(-6.290028433759329e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289348+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824194587e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756958852+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750736396e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713778851e-06+0j) [Y5 X6 X10 Y11] +
(0.0201759217235354+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.01128519020084084+0j) [Y5 X6 X11 Y12] +
(-4.55656921851811e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337824194587e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756958852+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750736396e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713778851e-06+0j) [Y5 Y6 Y10 Y11] +
(0.0201759217235354+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.01128519020084084+0j) [Y5 Y6 X11 X12] +
(-4.55656921851811e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694565+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.226048498862674e-05+0j) [Y5 Z6 Y7] +
(-1.8818501831631959e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449080895562e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613865+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613865+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913885102888e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730235+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.00796088072592154+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978286759633e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155945533e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731887369542e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561344+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731887369542e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561344+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179576+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736696669e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179576+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736696669e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928900514e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775801987e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622039105318e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156195+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529027+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.02563723829602683+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312896521674e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847253+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344676459781e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.041718813839821775+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817865196735e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638312+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362216178788e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719759+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579269282e-06+0j) [Y5 Y7] +
(-6.290028433759329e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289348+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145645+0j) [Z5] +
(0.010960074940542594+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468368349273e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542594+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468368349273e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065545+0j) [Z5 Z6] +
(0.01826683486937551+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174774073516e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01826683486937551+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174774073516e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040755+0j) [Z5 Z7] +
(0.15676396176430998+0j) [Z5 Z8] +
(-1.8163031700035276e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031700035276e-06+0j) [Z5 Y10 Z11 Y12] +
(0.1425799771248576+0j) [Z5 Z10] +
(1.8782101248144929e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101248144929e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237605+0j) [Z5 Z11] +
(0.1521504070886904+0j) [Z5 Z12] +
(0.11383573679388652+0j) [Z5 Z13] +
(-0.013873381748426119+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786328+0j) [X6 X7 Y10 Y11] +
(-1.0358477600720638e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477600720638e-06+0j) [X6 X7 X11 X12] +
(-0.01736611899465132+0j) [X6 X7 Y12 Y13] +
(0.013873381748426119+0j) [X6 Y7 Y8 X9] +
(0.017825140995786328+0j) [X6 Y7 Y10 X11] +
(1.0358477600720638e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477600720638e-06+0j) [X6 Y7 Y11 X12] +
(0.01736611899465132+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110879+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393505485725e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110879+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393505485725e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918685+0j) [X6 Z7 Z8 Z9 X10] +
(3.3131455002865665e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.3131455002865665e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848154+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844454+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671482+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231172972+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231172972+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860073128037e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559571195e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373849047736e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283487611686e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345653+0j) [X6 Z7 Z8 X10] +
(-3.277483195935763e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.03010462314345674+0j) [X6 Z7 Z9 X10] +
(-3.6102971309906202e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143863+0j) [X6 Z8 Z9 X10] +
(-3.7696594523791916e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426119+0j) [Y6 X7 X8 Y9] +
(0.017825140995786328+0j) [Y6 X7 X10 Y11] +
(1.0358477600720638e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477600720638e-06+0j) [Y6 X7 X11 Y12] +
(0.01736611899465132+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426119+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786328+0j) [Y6 Y7 X10 X11] +
(-1.0358477600720638e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477600720638e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.01736611899465132+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110879+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393505485725e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110879+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393505485725e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918685+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.3131455002865665e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.3131455002865665e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848154+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844454+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671482+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231172972+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231172972+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860073128037e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559571195e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373849047736e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283487611686e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345653+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195935763e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.03010462314345674+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971309906202e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143863+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594523791916e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.309686298861546+0j) [Z6] +
(0.030787505389143863+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594523791916e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143863+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594523791916e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270213+0j) [Z6 Z7] +
(0.16756653265461274+0j) [Z6 Z8] +
(0.18143991440303886+0j) [Z6 Z9] +
(-1.8551201217377704e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201217377704e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682675+0j) [Z6 Z10] +
(-2.890967881809834e-06+0j) [Z6 X11 Z12 X13] +
(-2.890967881809834e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261307+0j) [Z6 Z11] +
(0.13401715261963698+0j) [Z6 Z12] +
(0.15138327161428827+0j) [Z6 Z13] +
(-0.0002921986261110879+0j) [X7 X8 Y9 Y10] +
(3.3281393505485725e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110879+0j) [X7 Y8 Y9 X10] +
(-3.3281393505485725e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131455002865665e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231172972+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131455002865665e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231172972+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918694+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671482+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844454+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086007312803e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559571195e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283487611686e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848154+0j) [X7 Z8 Z9 X11] +
(-6.524373849047736e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.03010462314345674+0j) [X7 Z8 Z10 X11] +
(-3.6102971309906202e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345653+0j) [X7 Z9 Z10 X11] +
(-3.277483195935763e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110879+0j) [Y7 X8 X9 Y10] +
(-3.3281393505485725e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110879+0j) [Y7 Y8 X9 X10] +
(3.3281393505485725e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131455002865665e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231172972+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131455002865665e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231172972+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918694+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671482+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844454+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086007312803e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559571195e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283487611686e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848154+0j) [Y7 Z8 Z9 Y11] +
(-6.524373849047736e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.03010462314345674+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971309906202e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345653+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195935763e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615463+0j) [Z7] +
(0.18143991440303886+0j) [Z7 Z8] +
(0.16756653265461274+0j) [Z7 Z9] +
(-2.890967881809834e-06+0j) [Z7 X10 Z11 X12] +
(-2.890967881809834e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261307+0j) [Z7 Z10] +
(-1.8551201217377704e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201217377704e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682675+0j) [Z7 Z11] +
(0.15138327161428827+0j) [Z7 Z12] +
(0.13401715261963698+0j) [Z7 Z13] +
(-0.009560705729135964+0j) [X8 X9 Y10 Y11] +
(6.628614202099348e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614202099348e-07+0j) [X8 X9 X11 X12] +
(-0.00608782248056185+0j) [X8 X9 Y12 Y13] +
(0.009560705729135964+0j) [X8 Y9 Y10 X11] +
(-6.628614202099348e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614202099348e-07+0j) [X8 Y9 Y11 X12] +
(0.00608782248056185+0j) [X8 Y9 Y12 X13] +
(0.009560705729135964+0j) [Y8 X9 X10 Y11] +
(-6.628614202099348e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614202099348e-07+0j) [Y8 X9 X11 Y12] +
(0.00608782248056185+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135964+0j) [Y8 Y9 X10 X11] +
(6.628614202099348e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614202099348e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.00608782248056185+0j) [Y8 Y9 X12 X13] +
(1.3693525634718189+0j) [Z8] +
(-1.5973171979749825e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171979749825e-06+0j) [Z8 Y10 Z11 Y12] +
(0.1376687264585257+0j) [Z8 Z10] +
(-9.344557777650475e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557777650475e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766169+0j) [Z8 Z11] +
(0.14973486803496922+0j) [Z8 Z12] +
(0.15582269051553105+0j) [Z8 Z13] +
(1.3693525634718193+0j) [Z9] +
(-9.344557777650475e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557777650475e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766169+0j) [Z9 Z10] +
(-1.5973171979749825e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171979749825e-06+0j) [Z9 Y11 Z12 Y13] +
(0.1376687264585257+0j) [Z9 Z11] +
(0.15582269051553105+0j) [Z9 Z12] +
(0.14973486803496922+0j) [Z9 Z13] +
(-0.028685183716105987+0j) [X10 X11 Y12 Y13] +
(0.028685183716105987+0j) [X10 Y11 Y12 X13] +
(-1.0722312158332535e-05+0j) [X10 Z11 X12] +
(7.954413176687615e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372818577e-06+0j) [X10 X12] +
(0.028685183716105987+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105987+0j) [Y10 Y11 X12 X13] +
(-1.0722312158332535e-05+0j) [Y10 Z11 Y12] +
(7.954413176687615e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372818577e-06+0j) [Y10 Y12] +
(0.7829661725950191+0j) [Z10] +
(-8.194261372818577e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372818577e-06+0j) [Z10 Y11 Z12 Y13] +
(0.1492635514738891+0j) [Z10 Z11] +
(0.11270386920332208+0j) [Z10 Z12] +
(0.14138905291942808+0j) [Z10 Z13] +
(-1.0722312158332538e-05+0j) [X11 Z12 X13] +
(7.954413176687615e-06+0j) [X11 X13] +
(-1.0722312158332538e-05+0j) [Y11 Z12 Y13] +
(7.954413176687615e-06+0j) [Y11 Y13] +
(0.7829661725950191+0j) [Z11] +
(0.14138905291942808+0j) [Z11 Z12] +
(0.11270386920332208+0j) [Z11 Z13] +
(0.8084581961720472+0j) [Z12] +
(0.15435748657223622+0j) [Z12 Z13] +
(0.8084581961720472+0j) [Z13]
  (-46.46390678868897) [I0]
+ (0.7829661725950205) [Z10]
+ (0.7829661725950205) [Z11]
+ (0.8084581961720467) [Z13]
+ (0.8084581961720473) [Z12]
+ (1.2034402289145651) [Z4]
+ (1.2034402289145658) [Z5]
+ (1.3096862988615405) [Z6]
+ (1.3096862988615405) [Z7]
+ (1.3693525634718196) [Z8]
+ (1.36935256347182) [Z9]
+ (1.6538942226831697) [Z3]
+ (1.6538942226831699) [Z2]
+ (-8.194261371469149e-06) [Y10 Y12]
+ (-8.194261371469149e-06) [X10 X12]
+ (-1.8540608581889845e-06) [Y5 Y7]
+ (-1.8540608581889845e-06) [X5 X7]
+ (-7.764994118060154e-07) [Y3 Y5]
+ (-7.764994118060154e-07) [X3 X5]
+ (-5.929765817225472e-07) [Y4 Y6]
+ (-5.929765817225472e-07) [X4 X6]
+ (1.602116740727792e-06) [Y2 Y4]
+ (1.602116740727792e-06) [X2 X4]
+ (7.954413175679157e-06) [Y11 Y13]
+ (7.954413175679157e-06) [X11 X13]
+ (0.003276971931231677) [Y1 Y3]
+ (0.003276971931231677) [X1 X3]
+ (0.10433064780651427) [Y0 Y2]
+ (0.10433064780651427) [X0 X2]
+ (0.11270386920332201) [Z10 Z12]
+ (0.11270386920332201) [Z11 Z13]
+ (0.1138357367938865) [Z4 Z12]
+ (0.1138357367938865) [Z5 Z13]
+ (0.11952438964682657) [Z6 Z10]
+ (0.11952438964682657) [Z7 Z11]
+ (0.12489990917237603) [Z4 Z10]
+ (0.12489990917237603) [Z5 Z11]
+ (0.12495807739503227) [Z2 Z4]
+ (0.12495807739503227) [Z3 Z5]
+ (0.12799502492468404) [Z2 Z10]
+ (0.12799502492468404) [Z3 Z11]
+ (0.13401715261963665) [Z6 Z12]
+ (0.13401715261963665) [Z7 Z13]
+ (0.1370119167404075) [Z4 Z6]
+ (0.1370119167404075) [Z5 Z7]
+ (0.1373495306426132) [Z6 Z11]
+ (0.1373495306426132) [Z7 Z10]
+ (0.1373910476268321) [Z2 Z6]
+ (0.1373910476268321) [Z3 Z7]
+ (0.13766872645852585) [Z8 Z10]
+ (0.13766872645852585) [Z9 Z11]
+ (0.14011289865354787) [Z2 Z12]
+ (0.14011289865354787) [Z3 Z13]
+ (0.14138905291942785) [Z10 Z13]
+ (0.14138905291942785) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.14899430575065553) [Z4 Z7]
+ (0.14899430575065553) [Z5 Z6]
+ (0.14926355147388876) [Z10 Z11]
+ (0.14960702684445315) [Z4 Z8]
+ (0.14960702684445315) [Z5 Z9]
+ (0.14973486803496908) [Z8 Z12]
+ (0.14973486803496908) [Z9 Z13]
+ (0.15071408121008298) [Z2 Z8]
+ (0.15071408121008298) [Z3 Z9]
+ (0.1513832716142881) [Z6 Z13]
+ (0.1513832716142881) [Z7 Z12]
+ (0.15215040708869035) [Z4 Z13]
+ (0.15215040708869035) [Z5 Z12]
+ (0.15337968243314137) [Z2 Z11]
+ (0.15337968243314137) [Z3 Z10]
+ (0.1543574865722359) [Z12 Z13]
+ (0.1556901067175243) [Z2 Z13]
+ (0.1556901067175243) [Z3 Z12]
+ (0.1558226905155309) [Z8 Z13]
+ (0.1558226905155309) [Z9 Z12]
+ (0.15676396176431012) [Z4 Z9]
+ (0.15676396176431012) [Z5 Z8]
+ (0.15755314797985678) [Z4 Z5]
+ (0.1607976453483857) [Z2 Z5]
+ (0.1607976453483857) [Z3 Z4]
+ (0.16853486561579928) [Z2 Z7]
+ (0.16853486561579928) [Z3 Z6]
+ (0.18143991440303858) [Z6 Z9]
+ (0.18143991440303858) [Z7 Z8]
+ (0.18189085790751355) [Z2 Z3]
+ (0.18690820476912567) [Z2 Z9]
+ (0.18690820476912567) [Z3 Z8]
+ (0.19299723935364246) [Z0 Z10]
+ (0.19299723935364246) [Z1 Z11]
+ (0.1939253461327014) [Z6 Z7]
+ (0.1966177089034217) [Z0 Z4]
+ (0.1966177089034217) [Z1 Z5]
+ (0.19936354537360854) [Z0 Z5]
+ (0.19936354537360854) [Z1 Z4]
+ (0.20072866460441774) [Z0 Z11]
+ (0.20072866460441774) [Z1 Z10]
+ (0.2110265984979148) [Z0 Z12]
+ (0.2110265984979148) [Z1 Z13]
+ (0.21631037498631775) [Z0 Z13]
+ (0.21631037498631775) [Z1 Z12]
+ (0.22003977334376112) [Z8 Z9]
+ (0.23671080783830437) [Z0 Z2]
+ (0.23671080783830437) [Z1 Z3]
+ (0.24164663936017158) [Z0 Z6]
+ (0.24164663936017158) [Z1 Z7]
+ (0.2485348337131421) [Z0 Z7]
+ (0.2485348337131421) [Z1 Z6]
+ (0.2512944567459171) [Z0 Z3]
+ (0.2512944567459171) [Z1 Z2]
+ (0.2723251830660569) [Z0 Z8]
+ (0.2723251830660569) [Z1 Z9]
+ (0.2788345442672342) [Z0 Z9]
+ (0.2788345442672342) [Z1 Z8]
+ (1.1861763734860498) [Z0 Z1]
+ (-1.2260484990288963e-05) [Y5 Z6 Y7]
+ (-1.2260484990288963e-05) [X5 Z6 X7]
+ (-1.2260484990288962e-05) [Y4 Z5 Y6]
+ (-1.2260484990288962e-05) [X4 Z5 X6]
+ (-1.07223121576198e-05) [Y11 Z12 Y13]
+ (-1.07223121576198e-05) [X11 Z12 X13]
+ (-1.0722312157619796e-05) [Y10 Z11 Y12]
+ (-1.0722312157619796e-05) [X10 Z11 X12]
+ (-3.887051674135996e-06) [Y2 Z3 Y4]
+ (-3.887051674135996e-06) [X2 Z3 X4]
+ (-3.887051674135996e-06) [Y3 Z4 Y5]
+ (-3.887051674135996e-06) [X3 Z4 X5]
+ (0.12507032579772098) [Y1 Z2 Y3]
+ (0.12507032579772098) [X1 Z2 X3]
+ (0.12507032579772104) [Y0 Z1 Y2]
+ (0.12507032579772104) [X0 Z1 X2]
+ (-0.03831467029480384) [Y4 Y5 X12 X13]
+ (-0.03831467029480384) [X4 X5 Y12 Y13]
+ (-0.03619412355904269) [Y2 Y3 X8 X9]
+ (-0.03619412355904269) [X2 X3 Y8 Y9]
+ (-0.035839567953353434) [Y2 Y3 X4 X5]
+ (-0.035839567953353434) [X2 X3 Y4 Y5]
+ (-0.031143817988967183) [Y2 Y3 X6 X7]
+ (-0.031143817988967183) [X2 X3 Y6 Y7]
+ (-0.028685183716105837) [Y10 Y11 X12 X13]
+ (-0.028685183716105837) [X10 X11 Y12 Y13]
+ (-0.025996177598021086) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021086) [X3 Z4 Z5 X7]
+ (-0.025384657508457333) [Y2 Y3 X10 X11]
+ (-0.025384657508457333) [X2 X3 Y10 Y11]
+ (-0.019028242443847133) [Y3 Y4 X11 X12]
+ (-0.019028242443847133) [X3 X4 Y11 Y12]
+ (-0.017825140995786633) [Y6 Y7 X10 X11]
+ (-0.017825140995786633) [X6 X7 Y10 Y11]
+ (-0.017680067952481445) [Y4 Y5 X10 X11]
+ (-0.017680067952481445) [X4 X5 Y10 Y11]
+ (-0.01736611899465144) [Y6 Y7 X12 X13]
+ (-0.01736611899465144) [X6 X7 Y12 Y13]
+ (-0.01557720806397643) [Y2 Y3 X12 X13]
+ (-0.01557720806397643) [X2 X3 Y12 Y13]
+ (-0.014583648907612727) [Y0 Y1 X2 X3]
+ (-0.014583648907612727) [X0 X1 Y2 Y3]
+ (-0.013873381748426058) [Y6 Y7 X8 X9]
+ (-0.013873381748426058) [X6 X7 Y8 Y9]
+ (-0.011982389010248017) [Y4 Y5 X6 X7]
+ (-0.011982389010248017) [X4 X5 Y6 Y7]
+ (-0.011285190200840964) [Y5 X6 X11 Y12]
+ (-0.011285190200840964) [X5 Y6 Y11 X12]
+ (-0.009560705729135916) [Y8 Y9 X10 X11]
+ (-0.009560705729135916) [X8 X9 Y10 Y11]
+ (-0.008125251921381046) [Y1 X2 X8 Y9]
+ (-0.008125251921381046) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381046) [X1 X2 X8 X9]
+ (-0.008125251921381046) [X1 Y2 Y8 X9]
+ (-0.007731425250775282) [Y0 Y1 X10 X11]
+ (-0.007731425250775282) [X0 X1 Y10 Y11]
+ (-0.00715693491985696) [Y4 Y5 X8 X9]
+ (-0.00715693491985696) [X4 X5 Y8 Y9]
+ (-0.006509361201177245) [Y0 Y1 X8 X9]
+ (-0.006509361201177245) [X0 X1 Y8 Y9]
+ (-0.006087822480561841) [Y8 Y9 X12 X13]
+ (-0.006087822480561841) [X8 X9 Y12 Y13]
+ (-0.005283776488402947) [Y0 Y1 X12 X13]
+ (-0.005283776488402947) [X0 X1 Y12 Y13]
+ (-0.005143391768825151) [Y3 X4 X5 Y6]
+ (-0.005143391768825151) [X3 Y4 Y5 X6]
+ (-0.004684903388155195) [Y1 X2 X6 Y7]
+ (-0.004684903388155195) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155195) [X1 X2 X6 X7]
+ (-0.004684903388155195) [X1 Y2 Y6 X7]
+ (-0.004575007626639198) [Y1 X2 X12 Y13]
+ (-0.004575007626639198) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639198) [X1 X2 X12 X13]
+ (-0.004575007626639198) [X1 Y2 Y12 X13]
+ (-0.00442485544944187) [Y1 X2 X4 Y5]
+ (-0.00442485544944187) [Y1 Y2 Y4 Y5]
+ (-0.00442485544944187) [X1 X2 X4 X5]
+ (-0.00442485544944187) [X1 Y2 Y4 X5]
+ (-0.003479511890334435) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334435) [X2 Z3 Z5 X6]
+ (-0.003479511890334435) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334435) [X3 Z4 Z6 X7]
+ (-0.0027458364701868215) [Y0 Y1 X4 X5]
+ (-0.0027458364701868215) [X0 X1 Y4 Y5]
+ (-0.0017992194936630422) [Y1 X2 X10 Y11]
+ (-0.0017992194936630422) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630422) [X1 X2 X10 X11]
+ (-0.0017992194936630422) [X1 Y2 Y10 X11]
+ (-0.0002921986261110034) [Y7 Y8 X9 X10]
+ (-0.0002921986261110034) [X7 X8 Y9 Y10]
+ (-8.194261371469149e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371469149e-06) [Z10 X11 Z12 X13]
+ (-7.80170749967098e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170749967098e-06) [X2 Z3 X4 Z11]
+ (-7.80170749967098e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170749967098e-06) [X3 Z4 X5 Z10]
+ (-4.64305106796584e-06) [Y3 X4 X10 Y11]
+ (-4.64305106796584e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106796584e-06) [X3 X4 X10 X11]
+ (-4.64305106796584e-06) [X3 Y4 Y10 X11]
+ (-4.5888551553664225e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551553664225e-06) [X4 Z5 X6 Z13]
+ (-4.5888551553664225e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551553664225e-06) [X5 Z6 X7 Z12]
+ (-4.556569217556712e-06) [Y5 X6 X12 Y13]
+ (-4.556569217556712e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217556712e-06) [X5 X6 X12 X13]
+ (-4.556569217556712e-06) [X5 Y6 Y12 X13]
+ (-3.694513293960475e-06) [Y4 X5 X11 Y12]
+ (-3.694513293960475e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513293960475e-06) [X4 X5 X11 X12]
+ (-3.694513293960475e-06) [X4 Y5 Y11 X12]
+ (-3.3440815569546653e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815569546653e-06) [Z0 X5 Z6 X7]
+ (-3.3440815569546653e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815569546653e-06) [Z1 X4 Z5 X6]
+ (-3.1586564317051405e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564317051405e-06) [X2 Z3 X4 Z10]
+ (-3.1586564317051405e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564317051405e-06) [X3 Z4 X5 Z11]
+ (-3.099349244038186e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349244038186e-06) [Z0 X4 Z5 X6]
+ (-3.099349244038186e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349244038186e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816395164e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816395164e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816395164e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816395164e-06) [Z7 X10 Z11 X12]
+ (-2.1776646047687182e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646047687182e-06) [Z0 X10 Z11 X12]
+ (-2.1776646047687182e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646047687182e-06) [Z1 X11 Z12 X13]
+ (-1.881850183461622e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183461622e-06) [X4 Z5 X6 Z9]
+ (-1.881850183461622e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183461622e-06) [X5 Z6 X7 Z8]
+ (-1.85512012125703e-06) [Z6 Y10 Z11 Y12]
+ (-1.85512012125703e-06) [Z6 X10 Z11 X12]
+ (-1.85512012125703e-06) [Z7 Y11 Z12 Y13]
+ (-1.85512012125703e-06) [Z7 X11 Z12 X13]
+ (-1.8540608581889845e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608581889845e-06) [X4 Z5 X6 Z7]
+ (-1.8163031694444511e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031694444511e-06) [Z4 X11 Z12 X13]
+ (-1.8163031694444511e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031694444511e-06) [Z5 X10 Z11 X12]
+ (-1.6923978284623824e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978284623824e-06) [X4 Z5 X6 Z10]
+ (-1.6923978284623824e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978284623824e-06) [X5 Z6 X7 Z11]
+ (-1.6148794136617404e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794136617404e-06) [Z0 X11 Z12 X13]
+ (-1.6148794136617404e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794136617404e-06) [Z1 X10 Z11 X12]
+ (-1.597317197618284e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197618284e-06) [Z8 X10 Z11 X12]
+ (-1.597317197618284e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197618284e-06) [Z9 X11 Z12 X13]
+ (-1.4548424493173726e-06) [Y3 X4 X6 Y7]
+ (-1.4548424493173726e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424493173726e-06) [X3 X4 X6 X7]
+ (-1.4548424493173726e-06) [X3 Y4 Y6 X7]
+ (-1.3980449083127104e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449083127104e-06) [X4 Z5 X6 Z8]
+ (-1.3980449083127104e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449083127104e-06) [X5 Z6 X7 Z9]
+ (-1.1954890102007877e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890102007877e-06) [X2 Z3 X4 Z7]
+ (-1.1954890102007877e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890102007877e-06) [X3 Z4 X5 Z6]
+ (-1.190850808546933e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808546933e-06) [Z0 X3 Z4 X5]
+ (-1.190850808546933e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808546933e-06) [Z1 X2 Z3 X4]
+ (-1.1708301371275672e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301371275672e-06) [Z2 X5 Z6 X7]
+ (-1.1708301371275672e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301371275672e-06) [Z3 X4 Z5 X6]
+ (-1.0632283421634794e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283421634794e-06) [Z2 X10 Z11 X12]
+ (-1.0632283421634794e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283421634794e-06) [Z3 X11 Z12 X13]
+ (-1.0358477603824863e-06) [Y6 X7 X11 Y12]
+ (-1.0358477603824863e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477603824863e-06) [X6 X7 X11 X12]
+ (-1.0358477603824863e-06) [X6 Y7 Y11 X12]
+ (-9.509249753191071e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249753191071e-07) [Z2 X4 Z5 X6]
+ (-9.509249753191071e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249753191071e-07) [Z3 X5 Z6 X7]
+ (-9.34455777509894e-07) [Z8 Y11 Z12 Y13]
+ (-9.34455777509894e-07) [Z8 X11 Z12 X13]
+ (-9.34455777509894e-07) [Z9 Y10 Z11 Y12]
+ (-9.34455777509894e-07) [Z9 X10 Z11 X12]
+ (-8.337746755908544e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755908544e-07) [Z0 X2 Z3 X4]
+ (-8.337746755908544e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755908544e-07) [Z1 X3 Z4 X5]
+ (-7.956895373512573e-07) [Y3 X4 X8 Y9]
+ (-7.956895373512573e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373512573e-07) [X3 X4 X8 X9]
+ (-7.956895373512573e-07) [X3 Y4 Y8 X9]
+ (-7.764994118060153e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118060153e-07) [X2 Z3 X4 Z5]
+ (-5.929765817225472e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765817225472e-07) [Z4 X5 Z6 X7]
+ (-5.770052996053503e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996053503e-07) [X2 Z3 X4 Z9]
+ (-5.770052996053503e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996053503e-07) [X3 Z4 X5 Z8]
+ (-5.47164774426823e-07) [Y1 Y2 X11 X12]
+ (-5.47164774426823e-07) [X1 X2 Y11 Y12]
+ (-4.838052751489113e-07) [Y5 X6 X8 Y9]
+ (-4.838052751489113e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751489113e-07) [X5 X6 X8 X9]
+ (-4.838052751489113e-07) [X5 Y6 Y8 X9]
+ (-3.5707613295607866e-07) [Y0 X1 X3 Y4]
+ (-3.5707613295607866e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613295607866e-07) [X0 X1 X3 X4]
+ (-3.5707613295607866e-07) [X0 Y1 Y3 X4]
+ (-2.4473231291647977e-07) [Y0 X1 X5 Y6]
+ (-2.4473231291647977e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231291647977e-07) [X0 X1 X5 X6]
+ (-2.4473231291647977e-07) [X0 Y1 Y5 X6]
+ (-2.199051618084603e-07) [Y2 X3 X5 Y6]
+ (-2.199051618084603e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618084603e-07) [X2 X3 X5 X6]
+ (-2.199051618084603e-07) [X2 Y3 Y5 X6]
+ (-1.933241277318008e-07) [Y1 X2 X3 Y4]
+ (-1.933241277318008e-07) [X1 Y2 Y3 X4]
+ (-1.291969486323071e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486323071e-07) [X1 Z2 Z3 X5]
+ (1.7379332625872718e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332625872718e-07) [X0 Z1 Z3 X4]
+ (1.7379332625872718e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332625872718e-07) [X1 Z2 Z4 X5]
+ (1.933241277318008e-07) [Y1 Y2 X3 X4]
+ (1.933241277318008e-07) [X1 X2 Y3 Y4]
+ (2.1868423774590697e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423774590697e-07) [X2 Z3 X4 Z8]
+ (2.1868423774590697e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423774590697e-07) [X3 Z4 X5 Z9]
+ (2.5935343911658507e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343911658507e-07) [X2 Z3 X4 Z6]
+ (2.5935343911658507e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343911658507e-07) [X3 Z4 X5 Z7]
+ (3.606071868327634e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868327634e-07) [X0 Z1 Z2 X4]
+ (3.606071868327634e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868327634e-07) [X1 Z3 Z4 X5]
+ (5.47164774426823e-07) [Y1 X2 X11 Y12]
+ (5.47164774426823e-07) [X1 Y2 Y11 X12]
+ (5.627851911069778e-07) [Y0 X1 X11 Y12]
+ (5.627851911069778e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911069778e-07) [X0 X1 X11 X12]
+ (5.627851911069778e-07) [X0 Y1 Y11 X12]
+ (6.6286142010839e-07) [Y8 X9 X11 Y12]
+ (6.6286142010839e-07) [Y8 Y9 Y11 Y12]
+ (6.6286142010839e-07) [X8 X9 X11 X12]
+ (6.6286142010839e-07) [X8 Y9 Y11 X12]
+ (1.109440759206521e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759206521e-06) [Z2 X11 Z12 X13]
+ (1.109440759206521e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759206521e-06) [Z3 X10 Z11 X12]
+ (1.602116740727792e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740727792e-06) [Z2 X3 Z4 X5]
+ (1.8782101245160236e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101245160236e-06) [Z4 X10 Z11 X12]
+ (1.8782101245160236e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101245160236e-06) [Z5 X11 Z12 X13]
+ (2.172669101370001e-06) [Y2 X3 X11 Y12]
+ (2.172669101370001e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101370001e-06) [X2 X3 X11 X12]
+ (2.172669101370001e-06) [X2 Y3 Y11 X12]
+ (3.117447946483666e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946483666e-06) [X0 Z2 Z3 X4]
+ (3.5390541842206423e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541842206423e-06) [X2 Z3 X4 Z12]
+ (3.5390541842206423e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541842206423e-06) [X3 Z4 X5 Z13]
+ (4.2819138845121775e-06) [Y4 Z5 Y6 Z11]
+ (4.2819138845121775e-06) [X4 Z5 X6 Z11]
+ (4.2819138845121775e-06) [Y5 Z6 Y7 Z10]
+ (4.2819138845121775e-06) [X5 Z6 X7 Z10]
+ (5.275883121719287e-06) [Y3 X4 X12 Y13]
+ (5.275883121719287e-06) [Y3 Y4 Y12 Y13]
+ (5.275883121719287e-06) [X3 X4 X12 X13]
+ (5.275883121719287e-06) [X3 Y4 Y12 X13]
+ (5.97431171297456e-06) [Y5 X6 X10 Y11]
+ (5.97431171297456e-06) [Y5 Y6 Y10 Y11]
+ (5.97431171297456e-06) [X5 X6 X10 X11]
+ (5.97431171297456e-06) [X5 Y6 Y10 X11]
+ (7.954413175679157e-06) [Y10 Z11 Y12 Z13]
+ (7.954413175679157e-06) [X10 Z11 X12 Z13]
+ (8.814937305939929e-06) [Y2 Z3 Y4 Z13]
+ (8.814937305939929e-06) [X2 Z3 X4 Z13]
+ (8.814937305939929e-06) [Y3 Z4 Y5 Z12]
+ (8.814937305939929e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110034) [Y7 X8 X9 Y10]
+ (0.0002921986261110034) [X7 Y8 Y9 X10]
+ (0.000495676231491459) [Y2 Z4 Z5 Y6]
+ (0.000495676231491459) [X2 Z4 Z5 X6]
+ (0.001105903769189686) [Y0 Z1 Y2 Z5]
+ (0.001105903769189686) [X0 Z1 X2 Z5]
+ (0.001105903769189686) [Y1 Z2 Y3 Z4]
+ (0.001105903769189686) [X1 Z2 X3 Z4]
+ (0.0016638798784907156) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907156) [X2 Z3 Z4 X6]
+ (0.0016638798784907156) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907156) [X3 Z5 Z6 X7]
+ (0.001756070701841251) [Y0 Z1 Y2 Z11]
+ (0.001756070701841251) [X0 Z1 X2 Z11]
+ (0.001756070701841251) [Y1 Z2 Y3 Z10]
+ (0.001756070701841251) [X1 Z2 X3 Z10]
+ (0.0023262306231580836) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580836) [X0 Z1 X2 Z13]
+ (0.0023262306231580836) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580836) [X1 Z2 X3 Z12]
+ (0.0027458364701868215) [Y0 X1 X4 Y5]
+ (0.0027458364701868215) [X0 Y1 Y4 X5]
+ (0.0029297686747510525) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510525) [X0 Z1 X2 Z9]
+ (0.0029297686747510525) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510525) [X1 Z2 X3 Z8]
+ (0.0032769719312316773) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316773) [X0 Z1 X2 Z3]
+ (0.0033476175306661623) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661623) [X0 Z1 X2 Z7]
+ (0.0033476175306661623) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661623) [X1 Z2 X3 Z6]
+ (0.0035552901955042925) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042925) [X0 Z1 X2 Z10]
+ (0.0035552901955042925) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042925) [X1 Z2 X3 Z11]
+ (0.005143391768825151) [Y3 Y4 X5 X6]
+ (0.005143391768825151) [X3 X4 Y5 Y6]
+ (0.005283776488402947) [Y0 X1 X12 Y13]
+ (0.005283776488402947) [X0 Y1 Y12 X13]
+ (0.005530759218631556) [Y0 Z1 Y2 Z4]
+ (0.005530759218631556) [X0 Z1 X2 Z4]
+ (0.005530759218631556) [Y1 Z2 Y3 Z5]
+ (0.005530759218631556) [X1 Z2 X3 Z5]
+ (0.006087822480561841) [Y8 X9 X12 Y13]
+ (0.006087822480561841) [X8 Y9 Y12 X13]
+ (0.006509361201177245) [Y0 X1 X8 Y9]
+ (0.006509361201177245) [X0 Y1 Y8 X9]
+ (0.006901238249797282) [Y0 Z1 Y2 Z12]
+ (0.006901238249797282) [X0 Z1 X2 Z12]
+ (0.006901238249797282) [Y1 Z2 Y3 Z13]
+ (0.006901238249797282) [X1 Z2 X3 Z13]
+ (0.00715693491985696) [Y4 X5 X8 Y9]
+ (0.00715693491985696) [X4 Y5 Y8 X9]
+ (0.007731425250775282) [Y0 X1 X10 Y11]
+ (0.007731425250775282) [X0 Y1 Y10 X11]
+ (0.008032520918821357) [Y0 Z1 Y2 Z6]
+ (0.008032520918821357) [X0 Z1 X2 Z6]
+ (0.008032520918821357) [Y1 Z2 Y3 Z7]
+ (0.008032520918821357) [X1 Z2 X3 Z7]
+ (0.009560705729135916) [Y8 X9 X10 Y11]
+ (0.009560705729135916) [X8 Y9 Y10 X11]
+ (0.011055020596132099) [Y0 Z1 Y2 Z8]
+ (0.011055020596132099) [X0 Z1 X2 Z8]
+ (0.011055020596132099) [Y1 Z2 Y3 Z9]
+ (0.011055020596132099) [X1 Z2 X3 Z9]
+ (0.011285190200840964) [Y5 Y6 X11 X12]
+ (0.011285190200840964) [X5 X6 Y11 Y12]
+ (0.011307274008848255) [Y7 Z8 Z9 Y11]
+ (0.011307274008848255) [X7 Z8 Z9 X11]
+ (0.011982389010248017) [Y4 X5 X6 Y7]
+ (0.011982389010248017) [X4 Y5 Y6 X7]
+ (0.013873381748426058) [Y6 X7 X8 Y9]
+ (0.013873381748426058) [X6 Y7 Y8 X9]
+ (0.014583648907612727) [Y0 X1 X2 Y3]
+ (0.014583648907612727) [X0 Y1 Y2 X3]
+ (0.01557720806397643) [Y2 X3 X12 Y13]
+ (0.01557720806397643) [X2 Y3 Y12 X13]
+ (0.01736611899465144) [Y6 X7 X12 Y13]
+ (0.01736611899465144) [X6 Y7 Y12 X13]
+ (0.017680067952481445) [Y4 X5 X10 Y11]
+ (0.017680067952481445) [X4 Y5 Y10 X11]
+ (0.017825140995786633) [Y6 X7 X10 Y11]
+ (0.017825140995786633) [X6 Y7 Y10 X11]
+ (0.019028242443847133) [Y3 X4 X11 Y12]
+ (0.019028242443847133) [X3 Y4 Y11 X12]
+ (0.025384657508457333) [Y2 X3 X10 Y11]
+ (0.025384657508457333) [X2 Y3 Y10 X11]
+ (0.028685183716105837) [Y10 X11 X12 Y13]
+ (0.028685183716105837) [X10 Y11 Y12 X13]
+ (0.02981242451734597) [Y6 Z7 Z8 Y10]
+ (0.02981242451734597) [X6 Z7 Z8 X10]
+ (0.02981242451734597) [Y7 Z9 Z10 Y11]
+ (0.02981242451734597) [X7 Z9 Z10 X11]
+ (0.030104623143456972) [Y6 Z7 Z9 Y10]
+ (0.030104623143456972) [X6 Z7 Z9 X10]
+ (0.030104623143456972) [Y7 Z8 Z10 Y11]
+ (0.030104623143456972) [X7 Z8 Z10 X11]
+ (0.030787505389144005) [Y6 Z8 Z9 Y10]
+ (0.030787505389144005) [X6 Z8 Z9 X10]
+ (0.031143817988967183) [Y2 X3 X6 Y7]
+ (0.031143817988967183) [X2 Y3 Y6 X7]
+ (0.035839567953353434) [Y2 X3 X4 Y5]
+ (0.035839567953353434) [X2 Y3 Y4 X5]
+ (0.03619412355904269) [Y2 X3 X8 Y9]
+ (0.03619412355904269) [X2 Y3 Y8 X9]
+ (0.03831467029480384) [Y4 X5 X12 Y13]
+ (0.03831467029480384) [X4 Y5 Y12 X13]
+ (0.10433064780651427) [Z0 Y1 Z2 Y3]
+ (0.10433064780651427) [Z0 X1 Z2 X3]
+ (-0.12133276911042326) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042326) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042326) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042326) [X3 Z4 Z5 Z6 X7]
+ (3.2020768806201783e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768806201783e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768806201783e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768806201783e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564919013) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564919013) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564919013) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564919013) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329065) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329065) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329065) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329065) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273284) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273284) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273284) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273284) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021086) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021086) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646242) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646242) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646242) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646242) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173045) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173045) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173045) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173045) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997614026) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997614026) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997614026) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997614026) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997614026) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997614026) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997614026) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997614026) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819245) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819245) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819245) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819245) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688694) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688694) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688694) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688694) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688694) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688694) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688694) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688694) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381046) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381046) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832976) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832976) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832976) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832976) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826999) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826999) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826999) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826999) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017362) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017362) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017362) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017362) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825151) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825151) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825151) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825151) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155195) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155195) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776307) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776307) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639198) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639198) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.00442485544944187) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.00442485544944187) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840033) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840033) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840033) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840033) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901256) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901256) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901256) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901256) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025511) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025511) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524728) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524728) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630422) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630422) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369707) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369707) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730612) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730612) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730612) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730612) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125459) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125459) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.000814531327095724) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.000814531327095724) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.000814531327095724) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.000814531327095724) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880588914e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880588914e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880588914e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880588914e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817863695714e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817863695714e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817863695714e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817863695714e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215003445e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215003445e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215003445e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215003445e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675101626e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675101626e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675101626e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675101626e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373847838546e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373847838546e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373847838546e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373847838546e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432296422e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432296422e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432296422e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432296422e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311712974559e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311712974559e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883121719287e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883121719287e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106796584e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106796584e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692175567126e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692175567126e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225266047e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225266047e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451179091e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451179091e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132939604754e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132939604754e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297129762979e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297129762979e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297129762979e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297129762979e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500028882e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500028882e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483194664943e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483194664943e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483194664943e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483194664943e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228347809664e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228347809664e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228347809664e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228347809664e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.15134631089368e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.15134631089368e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507108806816e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507108806816e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101370001e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101370001e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424493173726e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424493173726e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731885940885e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731885940885e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337827070234e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337827070234e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477603824863e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477603824863e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373512573e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373512573e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741136432e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741136432e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741136432e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741136432e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.6286142010839e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.6286142010839e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914136387e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914136387e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914136387e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914136387e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.41829157417201e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.41829157417201e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.41829157417201e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.41829157417201e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453081747817e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453081747817e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453081747817e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453081747817e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911069778e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911069778e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624320778e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624320778e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624320778e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624320778e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624320778e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624320778e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624320778e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624320778e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751489113e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751489113e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613295607866e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613295607866e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350980363e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350980363e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265653833786e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265653833786e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265653833786e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265653833786e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231291647977e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231291647977e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289481862894e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289481862894e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289481862894e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289481862894e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516180846027e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516180846027e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241277318008e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241277318008e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241277318008e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241277318008e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915766205e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915766205e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915766205e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915766205e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539178561086e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539178561086e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539178561086e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539178561086e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781482860907e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781482860907e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781482860907e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781482860907e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781482860907e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781482860907e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781482860907e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781482860907e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781482860907e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781482860907e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781482860907e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781482860907e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486323071e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486323071e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598596947e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598596947e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598596947e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598596947e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598596947e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598596947e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598596947e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598596947e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446593886144e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446593886144e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446593886144e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446593886144e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310135924823e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310135924823e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310135924823e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310135924823e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209157662052e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209157662052e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209157662052e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209157662052e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516180846027e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516180846027e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231291647977e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231291647977e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599617787716e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599617787716e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599617787716e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599617787716e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350980363e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350980363e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613295607866e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613295607866e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751489113e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751489113e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911069778e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911069778e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.6286142010839e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.6286142010839e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373512573e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373512573e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651302075e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651302075e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651302075e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651302075e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477603824863e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477603824863e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337827070234e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337827070234e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216685454e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216685454e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216685454e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216685454e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731885940885e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731885940885e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424493173726e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424493173726e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101370001e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101370001e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507108806816e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507108806816e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946483666e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946483666e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.15134631089368e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.15134631089368e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500028882e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500028882e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312891846027e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312891846027e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132939604754e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132939604754e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559385246e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559385246e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692175567126e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692175567126e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106796584e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106796584e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883121719287e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883121719287e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311712974559e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311712974559e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110034) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110034) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110034) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110034) [X6 Z7 X8 X9 Z10 X11]
+ (0.000495676231491459) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.000495676231491459) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499077) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499077) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499077) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499077) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125459) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125459) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213828) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213828) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213828) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213828) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440443) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440443) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440443) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440443) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369707) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369707) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630422) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630422) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524728) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524728) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133929) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133929) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133929) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133929) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496517) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496517) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496517) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496517) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.00442485544944187) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.00442485544944187) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639198) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639198) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776307) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776307) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155195) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155195) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221701) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221701) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221701) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221701) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109625) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109625) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109625) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109625) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921615) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921615) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921615) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921615) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381046) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381046) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694675) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694675) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694675) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694675) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158443) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158443) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158443) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158443) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671534) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671534) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671534) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671534) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.01096007494054268) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.01096007494054268) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.01096007494054268) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.01096007494054268) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848256) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848256) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130818) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130818) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130818) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130818) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226542) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226542) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226542) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226542) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380144) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380144) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380144) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380144) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375657) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375657) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375657) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375657) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173040056) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173040056) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173040056) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173040056) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353564) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353564) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353564) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353564) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353564) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353564) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353564) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353564) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068834) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068834) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068834) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068834) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068834) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068834) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068834) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068834) [X3 Z4 X5 X10 Z11 X12]
+ (0.02510495713884458) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884458) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884458) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884458) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389144005) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389144005) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129799) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129799) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780795) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780795) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780795) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780795) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661384) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661384) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661384) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661384) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277927734152e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277927734152e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277927734152e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277927734152e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860063504753e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860063504753e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860063504733e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860063504733e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378318) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378318) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378319) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378319) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638313) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638313) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638313) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638313) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982179) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982179) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982179) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982179) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289316) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289316) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289316) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289316) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205296) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205296) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205296) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205296) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719748) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719748) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719748) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719748) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312483) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312483) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905474) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905474) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905474) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905474) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026838) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026838) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026838) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026838) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890897) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890897) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890897) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890897) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469303) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469303) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952896) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952896) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.0225284401960131) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0225284401960131) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600683) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600683) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600683) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600683) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251627) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251627) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847133) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847133) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942926) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942926) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942926) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942926) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917947) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917947) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226542) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226542) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460370472916206) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.01460370472916206) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173045) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173045) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819245) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840964) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840964) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962546) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962546) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847373) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847373) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847373) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847373) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791024095) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791024095) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832976) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832976) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561343) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561343) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017362) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017362) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109625) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109625) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840033) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840033) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328823) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328823) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328823) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328823) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235437) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235437) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235437) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235437) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025511) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025511) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806614) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806614) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806614) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806614) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524728) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524728) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524728) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524728) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696443) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696443) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696443) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696443) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696443) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696443) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696443) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696443) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569567775) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569567775) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355197) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730355197) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730355197) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730355197) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880588914e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880588914e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585304038744e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585304038744e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585304038744e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585304038744e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808793670818e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808793670818e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808793670818e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808793670818e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277432914e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277432914e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277432914e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277432914e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.08979946686501e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.08979946686501e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.08979946686501e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.08979946686501e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209667916507e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209667916507e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209667916507e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209667916507e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183234211e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.48185183234211e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183234211e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.48185183234211e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736169397e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736169397e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736169397e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736169397e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038159741e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038159741e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038159741e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038159741e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.72884314675085e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.72884314675085e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.72884314675085e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.72884314675085e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225266047e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225266047e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451179091e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451179091e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954289613073e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954289613073e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954289613073e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954289613073e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954289613073e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954289613073e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954289613073e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954289613073e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563201141596e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563201141596e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563201141596e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563201141596e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156041632617e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156041632617e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156041632617e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156041632617e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220975251677e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220975251677e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220975251677e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220975251677e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836305083e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836305083e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836305083e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836305083e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174765065812e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174765065812e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174765065812e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174765065812e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675267975e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675267975e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675267975e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675267975e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675267975e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675267975e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675267975e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675267975e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337827070234e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337827070234e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337827070234e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337827070234e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770289650169e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770289650169e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770289650169e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770289650169e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.86776510367927e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510367927e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510367927e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.86776510367927e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974342278e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974342278e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206674177e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206674177e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774426823e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774426823e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447180806256e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447180806256e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447180806256e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447180806256e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389676941964e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389676941964e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231088439135e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231088439135e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231088439135e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231088439135e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350980363e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350980363e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350980363e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350980363e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265653833786e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265653833786e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935979850185e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935979850185e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935979850185e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935979850185e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289481862894e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289481862894e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915766205e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915766205e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446593886145e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446593886145e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095955422e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095955422e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095955422e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095955422e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446593886145e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446593886145e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350663809407e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350663809407e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350663809407e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350663809407e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783557439653e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783557439653e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783557439653e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783557439653e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915766205e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915766205e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289481862894e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289481862894e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265653833786e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265653833786e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389676941964e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389676941964e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774426823e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774426823e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206674177e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206674177e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974342278e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974342278e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731885940887e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731885940887e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731885940887e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731885940887e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532433668818e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532433668818e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532433668818e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532433668818e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489513127232e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489513127232e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489513127232e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489513127232e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.74551839999629e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.74551839999629e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.74551839999629e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.74551839999629e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.74551839999629e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.74551839999629e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.74551839999629e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.74551839999629e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420188395207e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420188395207e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420188395207e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420188395207e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420188395207e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420188395207e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420188395207e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420188395207e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500028882e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500028882e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500028882e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500028882e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289184603e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289184603e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559385246e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559385246e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880588914e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880588914e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569567775) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569567775) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128841007) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128841007) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128841007) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128841007) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005477) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005477) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005477) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005477) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005477) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005477) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005477) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005477) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125458) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125458) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125458) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125458) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907672) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907672) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907672) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907672) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496819) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496819) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496819) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496819) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126936) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126936) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126936) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126936) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823377) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823377) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823377) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823377) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823377) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823377) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823377) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823377) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.0039898414566193075) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.0039898414566193075) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.0039898414566193075) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.0039898414566193075) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840033) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840033) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914311) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914311) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914311) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914311) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182564) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182564) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182564) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182564) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660408) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660408) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660408) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660408) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660408) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660408) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660408) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660408) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00524153538280386) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.00524153538280386) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.00524153538280386) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.00524153538280386) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076854) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076854) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076854) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076854) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109625) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109625) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839381) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839381) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839381) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839381) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017362) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017362) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960956) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960956) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960956) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960956) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561343) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561343) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832976) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832976) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791024095) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791024095) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962546) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962546) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840964) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840964) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819245) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173045) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173045) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.01460370472916206) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.01460370472916206) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226542) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226542) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917947) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917947) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847133) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847133) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251627) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251627) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129798) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129798) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156156) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156156) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156156) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156156) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702274) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702274) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702271) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702271) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0906514420703648) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703648) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703648) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703648) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863625) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863625) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863625) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863625) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634986) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634986) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634986) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634986) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214003) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214003) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214003) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214003) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312483) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312483) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366188) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366188) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366188) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366188) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383003) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088383003) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383003) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088383003) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469303) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469303) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929528964) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929528964) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013102) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013102) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314604) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314604) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314604) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314604) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898755) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898755) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898755) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898755) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917947) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917947) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917947) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917947) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831854) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831854) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831854) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831854) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962546) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962546) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962546) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962546) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209844) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209844) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209844) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209844) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454826) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454826) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454826) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454826) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454826) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454826) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454826) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454826) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791024095) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791024095) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791024095) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791024095) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776307) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776307) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336933) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336933) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285438) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285438) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285438) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285438) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178904) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178904) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328823) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328823) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235433) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235433) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369707) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369707) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124167) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124167) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168983) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168983) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168983) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168983) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024453) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024453) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487563) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487563) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029755963) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029755963) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355197) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730355197) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221155593e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221155593e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221155593e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221155593e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736169397e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736169397e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.15134631089368e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.15134631089368e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507108806816e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507108806816e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706474941e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706474941e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071162471e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071162471e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563201141596e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563201141596e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656082602e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656082602e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376505958502e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376505958502e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376505958502e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376505958502e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332101717294e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332101717294e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332101717294e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332101717294e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637197788374e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637197788374e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637197788374e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637197788374e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637197788374e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637197788374e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637197788374e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637197788374e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305984723311e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305984723311e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305984723311e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305984723311e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985112092e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985112092e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985112092e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985112092e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.86776510367927e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.86776510367927e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692463667569e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692463667569e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692463667569e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692463667569e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692463667569e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692463667569e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692463667569e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692463667569e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421565339e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421565339e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421565339e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421565339e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421565339e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421565339e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421565339e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421565339e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247520846409e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247520846409e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247520846409e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247520846409e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393081701277e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393081701277e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393081701277e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393081701277e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393081701277e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393081701277e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393081701277e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393081701277e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935979850185e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935979850185e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381541288085e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381541288085e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783557439653e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783557439653e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350663809407e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350663809407e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242153785e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773242153785e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242153785e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773242153785e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242153785e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773242153785e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773242153785e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773242153785e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253785856956e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253785856956e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253785856956e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253785856956e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716552782908e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716552782908e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716552782908e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716552782908e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350663809407e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350663809407e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282179164524e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282179164524e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282179164524e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282179164524e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493377363e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493377363e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493377363e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493377363e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783557439653e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783557439653e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943049540023e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943049540023e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943049540023e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943049540023e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381541288085e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381541288085e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935979850185e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935979850185e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506157600453e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506157600453e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506157600453e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506157600453e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506157600453e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506157600453e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506157600453e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506157600453e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978542082827e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978542082827e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978542082827e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978542082827e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150948874463e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150948874463e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150948874463e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150948874463e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974424590629e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974424590629e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974424590629e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974424590629e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974424590629e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974424590629e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974424590629e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974424590629e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.86776510367927e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.86776510367927e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656082602e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656082602e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563201141596e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563201141596e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071162471e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071162471e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765758811513e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765758811513e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011293645e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011293645e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011293645e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011293645e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706474941e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706474941e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507108806816e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507108806816e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.15134631089368e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.15134631089368e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671003524e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671003524e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671003524e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671003524e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736169397e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736169397e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721672838e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721672838e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721672838e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721672838e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327086126e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327086126e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327086126e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327086126e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501754835e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501754835e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501754835e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501754835e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988655883644e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988655883644e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988655883644e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988655883644e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717768586e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717768586e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717768586e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717768586e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347610961e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347610961e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579283531e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579283531e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579283531e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579283531e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112176605e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112176605e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112176605e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112176605e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730355197) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730355197) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389551854) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389551854) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389551854) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389551854) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029755963) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029755963) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569567775) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569567775) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569567775) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569567775) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487563) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487563) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908894) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908894) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908894) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908894) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024453) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024453) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730418) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730418) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730418) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730418) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124167) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124167) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369707) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369707) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.00244649715541585) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.00244649715541585) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.00244649715541585) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.00244649715541585) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235433) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235433) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328823) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328823) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178904) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178904) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336933) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336933) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776307) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776307) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278096) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278096) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278096) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278096) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226852) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226852) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226852) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226852) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409975) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409975) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409975) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409975) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.0107155084697968) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0107155084697968) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0107155084697968) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0107155084697968) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908977) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908977) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908977) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908977) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01460370472916206) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.01460370472916206) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.01460370472916206) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.01460370472916206) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0192995605793638) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0192995605793638) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861775) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861775) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.77595052682523e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.77595052682523e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505268252334e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505268252334e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0716503518100273) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100273) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002732) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002732) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251627) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251627) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831854) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831854) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209844) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209844) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770619) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770619) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770619) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770619) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766295) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766295) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766295) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766295) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285438) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285438) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219147) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219147) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219147) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219147) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.00244649715541585) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.00244649715541585) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939882) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939882) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939882) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939882) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015997) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015997) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587405) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587405) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587405) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587405) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587405) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587405) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587405) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587405) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124167) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124167) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124167) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124167) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538262) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538262) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538262) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538262) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538262) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538262) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538262) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538262) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562665) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562665) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562665) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562665) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452198622e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452198622e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071162471e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071162471e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071162471e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071162471e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946560826018e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946560826018e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946560826018e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946560826018e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.04449412971688e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.04449412971688e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.04449412971688e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.04449412971688e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229162732e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229162732e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229162732e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229162732e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.10551503618023e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.10551503618023e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.10551503618023e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.10551503618023e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212474361e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212474361e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212474361e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212474361e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413099727e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413099727e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974342278e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974342278e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657598411e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657598411e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657598411e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657598411e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206674177e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206674177e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389676941964e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389676941964e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531791934e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531791934e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531791934e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531791934e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458627106e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458627106e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998840690735e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998840690735e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998840690735e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998840690735e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317548406e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317548406e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317548406e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317548406e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929825022e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641929825022e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309317712914e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309317712914e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309317712914e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309317712914e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641929825022e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641929825022e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815412880854e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815412880854e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815412880854e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815412880854e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458627106e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458627106e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389676941964e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389676941964e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390398397e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390398397e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390398397e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390398397e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206674177e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206674177e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974342278e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974342278e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413099727e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413099727e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476486693146e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476486693146e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939575782548e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939575782548e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939575782548e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939575782548e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765758811517e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765758811517e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706474941e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706474941e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706474941e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706474941e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347610961e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347610961e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734299646e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734299646e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734299646e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734299646e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.58096036918779e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.58096036918779e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.58096036918779e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.58096036918779e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487563) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487563) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487563) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487563) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024453) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024453) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024453) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024453) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441944) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441944) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441944) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441944) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245329) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245329) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245329) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245329) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500461) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500461) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500461) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500461) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980205) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980205) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980205) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980205) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980205) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980205) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980205) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980205) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00244649715541585) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.00244649715541585) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285438) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285438) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369325) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369325) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369325) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369325) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046448) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046448) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046448) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046448) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209844) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209844) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831854) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831854) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251627) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251627) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861775) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861775) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015060806e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015060806e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009015060803e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015060803e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178904) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178904) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029755963) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029755963) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452198622e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452198622e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939575782548e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939575782548e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413099727e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413099727e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413099727e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413099727e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641929825022e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929825022e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929825022e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929825022e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458627106e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458627106e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458627106e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458627106e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476486693145e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476486693145e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939575782548e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939575782548e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755963) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755963) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178904) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178904) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352538) [I0]
+ (-0.18066792656583544) [Z7]
+ (-0.18066792656583536) [Z6]
+ (-0.15961432501810127) [Z4]
+ (-0.15961432501810124) [Z5]
+ (0.1741995615505545) [Z2]
+ (0.1741995615505547) [Z3]
+ (0.22757269005453296) [Z1]
+ (0.227572690054533) [Z0]
+ (-8.194261372192518e-06) [Y4 Y6]
+ (-8.194261372192518e-06) [X4 X6]
+ (7.954413176306334e-06) [Y5 Y7]
+ (7.954413176306334e-06) [X5 X7]
+ (0.11270386920332257) [Z4 Z6]
+ (0.11270386920332257) [Z5 Z7]
+ (0.11952438964682689) [Z0 Z4]
+ (0.11952438964682689) [Z1 Z5]
+ (0.13401715261963734) [Z0 Z6]
+ (0.13401715261963734) [Z1 Z7]
+ (0.13734953064261335) [Z0 Z5]
+ (0.13734953064261335) [Z1 Z4]
+ (0.1376687264585259) [Z2 Z4]
+ (0.1376687264585259) [Z3 Z5]
+ (0.1413890529194285) [Z4 Z7]
+ (0.1413890529194285) [Z5 Z6]
+ (0.14722943218766188) [Z2 Z5]
+ (0.14722943218766188) [Z3 Z4]
+ (0.14926355147388937) [Z4 Z5]
+ (0.14973486803496958) [Z2 Z6]
+ (0.14973486803496958) [Z3 Z7]
+ (0.1513832716142888) [Z0 Z7]
+ (0.1513832716142888) [Z1 Z6]
+ (0.15435748657223686) [Z6 Z7]
+ (0.15582269051553146) [Z2 Z7]
+ (0.15582269051553146) [Z3 Z6]
+ (0.1675665326546126) [Z0 Z2]
+ (0.1675665326546126) [Z1 Z3]
+ (0.18143991440303875) [Z0 Z3]
+ (0.18143991440303875) [Z1 Z2]
+ (0.19392534613270215) [Z0 Z1]
+ (0.2200397733437608) [Z2 Z3]
+ (-7.0378875107893684e-06) [Y5 Z6 Y7]
+ (-7.0378875107893684e-06) [X5 Z6 X7]
+ (-7.037887510789368e-06) [Y4 Z5 Y6]
+ (-7.037887510789368e-06) [X4 Z5 X6]
+ (-0.0286851837161059) [Y4 Y5 X6 X7]
+ (-0.0286851837161059) [X4 X5 Y6 Y7]
+ (-0.017825140995786446) [Y0 Y1 X4 X5]
+ (-0.017825140995786446) [X0 X1 Y4 Y5]
+ (-0.017366118994651458) [Y0 Y1 X6 X7]
+ (-0.017366118994651458) [X0 X1 Y6 Y7]
+ (-0.013873381748426148) [Y0 Y1 X2 X3]
+ (-0.013873381748426148) [X0 X1 Y2 Y3]
+ (-0.009560705729135964) [Y2 Y3 X4 X5]
+ (-0.009560705729135964) [X2 X3 Y4 Y5]
+ (-0.006087822480561887) [Y2 Y3 X6 X7]
+ (-0.006087822480561887) [X2 X3 Y6 Y7]
+ (-0.0002921986261110885) [Y1 Y2 X3 X4]
+ (-0.0002921986261110885) [X1 X2 Y3 Y4]
+ (-8.194261372192518e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372192518e-06) [Z4 X5 Z6 X7]
+ (-2.890967881595739e-06) [Z0 Y5 Z6 Y7]
+ (-2.890967881595739e-06) [Z0 X5 Z6 X7]
+ (-2.890967881595739e-06) [Z1 Y4 Z5 Y6]
+ (-2.890967881595739e-06) [Z1 X4 Z5 X6]
+ (-1.8551201215245492e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201215245492e-06) [Z0 X4 Z5 X6]
+ (-1.8551201215245492e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201215245492e-06) [Z1 X5 Z6 X7]
+ (-1.597317197777565e-06) [Z2 Y4 Z5 Y6]
+ (-1.597317197777565e-06) [Z2 X4 Z5 X6]
+ (-1.597317197777565e-06) [Z3 Y5 Z6 Y7]
+ (-1.597317197777565e-06) [Z3 X5 Z6 X7]
+ (-1.0358477600711897e-06) [Y0 X1 X5 Y6]
+ (-1.0358477600711897e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477600711897e-06) [X0 X1 X5 X6]
+ (-1.0358477600711897e-06) [X0 Y1 Y5 X6]
+ (-9.344557776014207e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557776014207e-07) [Z2 X5 Z6 X7]
+ (-9.344557776014207e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557776014207e-07) [Z3 X4 Z5 X6]
+ (6.628614201761443e-07) [Y2 X3 X5 Y6]
+ (6.628614201761443e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201761443e-07) [X2 X3 X5 X6]
+ (6.628614201761443e-07) [X2 Y3 Y5 X6]
+ (7.954413176306334e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176306334e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110885) [Y1 X2 X3 Y4]
+ (0.0002921986261110885) [X1 Y2 Y3 X4]
+ (0.006087822480561887) [Y2 X3 X6 Y7]
+ (0.006087822480561887) [X2 Y3 Y6 X7]
+ (0.009560705729135964) [Y2 X3 X4 Y5]
+ (0.009560705729135964) [X2 Y3 Y4 X5]
+ (0.011307274008848237) [Y1 Z2 Z3 Y5]
+ (0.011307274008848237) [X1 Z2 Z3 X5]
+ (0.013873381748426148) [Y0 X1 X2 Y3]
+ (0.013873381748426148) [X0 Y1 Y2 X3]
+ (0.017366118994651458) [Y0 X1 X6 Y7]
+ (0.017366118994651458) [X0 Y1 Y6 X7]
+ (0.017825140995786446) [Y0 X1 X4 Y5]
+ (0.017825140995786446) [X0 Y1 Y4 X5]
+ (0.0286851837161059) [Y4 X5 X6 Y7]
+ (0.0286851837161059) [X4 Y5 Y6 X7]
+ (0.029812424517345754) [Y0 Z1 Z2 Y4]
+ (0.029812424517345754) [X0 Z1 Z2 X4]
+ (0.029812424517345754) [Y1 Z3 Z4 Y5]
+ (0.029812424517345754) [X1 Z3 Z4 X5]
+ (0.030787505389143984) [Y0 Z2 Z3 Y4]
+ (0.030787505389143984) [X0 Z2 Z3 X4]
+ (0.04375263801065991) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801065991) [X0 Z1 Z2 Z3 X4]
+ (0.043752638010659914) [Y1 Z2 Z3 Z4 Y5]
+ (0.043752638010659914) [X1 Z2 Z3 Z4 X5]
+ (-0.014564531231172986) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172986) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172986) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172986) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848566257e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848566257e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848566257e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848566257e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594520407647e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594520407647e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.61029713065394e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.61029713065394e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.61029713065394e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.61029713065394e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455001314396e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455001314396e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277483195602462e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277483195602462e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277483195602462e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277483195602462e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283484348178e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283484348178e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283484348178e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283484348178e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477600711897e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477600711897e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201761443e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201761443e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393505147817e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393505147817e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393505147817e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393505147817e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201761443e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201761443e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477600711897e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477600711897e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455001314396e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455001314396e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559394756e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559394756e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611108856) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611108856) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611108856) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611108856) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671628) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671628) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671628) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671628) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848237) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848237) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844614) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844614) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844614) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844614) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143984) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143984) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549764285e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549764285e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396549764282e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549764282e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231172986) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172986) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594520407647e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594520407647e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393505147817e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505147817e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393505147817e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505147817e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.31314550013144e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.31314550013144e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.31314550013144e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.31314550013144e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559394756e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559394756e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172986) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172986) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
(-46.46390678868895+0j) [] +
(-0.014583648907612625+0j) [X0 X1 Y2 Y3] +
(-3.5707613289377187e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017313+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209803+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939577122988e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613289377187e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017312+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209803+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577122988e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186815+0j) [X0 X1 Y4 Y5] +
(-2.447323128731932e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104190514e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285377+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128731932e-07+0j) [X0 X1 X5 X6] +
(-7.867765104190514e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285373+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970523+0j) [X0 X1 Y6 Y7] +
(-7.735036880587501e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783554181135e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880587501e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783554181135e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177231+0j) [X0 X1 Y8 Y9] +
(-0.00773142525077526+0j) [X0 X1 Y10 Y11] +
(5.627851911439658e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911439658e-07+0j) [X0 X1 X11 X12] +
(-0.00528377648840294+0j) [X0 X1 Y12 Y13] +
(0.014583648907612625+0j) [X0 Y1 Y2 X3] +
(3.5707613289377187e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017313+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209803+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939577122988e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613289377187e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017312+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209803+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577122988e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186815+0j) [X0 Y1 Y4 X5] +
(2.447323128731932e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104190514e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285377+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128731932e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104190514e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285373+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970523+0j) [X0 Y1 Y6 X7] +
(7.735036880587501e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783554181135e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880587501e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783554181135e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177231+0j) [X0 Y1 Y8 X9] +
(0.00773142525077526+0j) [X0 Y1 Y10 X11] +
(-5.627851911439658e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911439658e-07+0j) [X0 Y1 Y11 X12] +
(0.00528377648840294+0j) [X0 Y1 Y12 X13] +
(0.1250703257977163+0j) [X0 Z1 X2] +
(-1.9332412771037803e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.002293956611352445+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123942+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714589251163e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771037803e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.002293956611352445+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123942+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714589251163e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231618+0j) [X0 Z1 X2 Z3] +
(-1.551053917618788e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376507657042e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770581+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480387602e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986493198e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676601+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00553075921863151+0j) [X0 Z1 X2 Z4] +
(-1.38077814803876e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.376739308465975e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587162+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.38077814803876e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.376739308465975e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587162+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896507+0j) [X0 Z1 X2 Z5] +
(0.005708495985960922+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332103171313e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253794463274e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076823+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305986020256e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821319+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005354+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773244506028e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005354+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244506028e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666108+0j) [X0 Z1 X2 Z7] +
(0.011055020596132031+0j) [X0 Z1 X2 Z8] +
(0.002929768674751001+0j) [X0 Z1 X2 Z9] +
(-6.41829157464426e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914647617e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504241+0j) [X0 Z1 X2 Z10] +
(-1.1076325599371486e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599371486e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018412067+0j) [X0 Z1 X2 Z11] +
(0.006901238249797223+0j) [X0 Z1 X2 Z12] +
(0.002326230623158027+0j) [X0 Z1 X2 Z13] +
(-3.5682475211638447e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939817+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716555371242e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288409843+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253793877233e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441859+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.5233896780272206e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178847+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199191065e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311866+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155211+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776288+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975374193e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660387+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464916172e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381029+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630346+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744706795e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624710785e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639197+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441859+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.5233896780272206e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178847+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199191065e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311866+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155211+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776288+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975374193e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660387+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464916172e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381029+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630346+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744706795e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624710785e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639197+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.202076878510505e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125413+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024423+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125413+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024423+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694864935834e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978540359136e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441896+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.684915095153227e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004446+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209154590184e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.09225061603563e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980136+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.09225061603563e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980136+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599610453856e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310131356678e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126902+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.0039898414566193014+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742645876e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823386+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823386+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453083077043e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363216887463e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536651729868e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562548+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.8394209154590186e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.0001940085702975689+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538236+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479097172e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446595688327e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369622+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696484+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265651575946e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209154590186e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.0001940085702975689+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538236+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289479097172e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446595688327e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369622+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696484+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265651575946e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.042743277013781944+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487632+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.850564192804681e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487632+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192804681e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025528+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182525+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(2.312094305281779e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282184183422e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839355+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425524599e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425524599e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803839+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914276+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907453+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287494159068e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328692+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303551523+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207106256e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422169207e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0032675138544235316+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328692+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303551523+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207106256e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422169207e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0032675138544235316+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336927+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413724025e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336927+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413724025e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002644+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002141361223101569+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046455+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245444+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219104+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219104+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009015301563e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476487301266e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658341897e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213160248e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.00153248352307306+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998843944724e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409987+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298118498e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278101+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.10551503724969e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226864+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.95607923005437e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001609531381721365+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221153446e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731754798062e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002462917007133906+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248909078+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.076732531910833e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.606071867706705e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496485+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389553708+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309316086426e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332622292967e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440404+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214168569+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402390533759e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651383+0j) [X0 X2] +
(3.117447945919852e-06+0j) [X0 Z2 Z3 X4] +
(0.045879470781297754+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861726+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453086924e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612625+0j) [Y0 X1 X2 Y3] +
(3.5707613289377187e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017313+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209803+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939577122988e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613289377187e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017312+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209803+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577122988e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186815+0j) [Y0 X1 X4 Y5] +
(2.447323128731932e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104190514e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285377+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128731932e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104190514e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285373+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970523+0j) [Y0 X1 X6 Y7] +
(7.735036880587501e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783554181135e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880587501e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783554181135e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177231+0j) [Y0 X1 X8 Y9] +
(0.00773142525077526+0j) [Y0 X1 X10 Y11] +
(-5.627851911439658e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911439658e-07+0j) [Y0 X1 X11 Y12] +
(0.00528377648840294+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612625+0j) [Y0 Y1 X2 X3] +
(-3.5707613289377187e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017313+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209803+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939577122988e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613289377187e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017312+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209803+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577122988e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186815+0j) [Y0 Y1 X4 X5] +
(-2.447323128731932e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104190514e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285377+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128731932e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104190514e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285373+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970523+0j) [Y0 Y1 X6 X7] +
(-7.735036880587501e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783554181135e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880587501e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783554181135e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177231+0j) [Y0 Y1 X8 X9] +
(-0.00773142525077526+0j) [Y0 Y1 X10 X11] +
(5.627851911439658e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911439658e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.00528377648840294+0j) [Y0 Y1 X12 X13] +
(-3.5682475211638447e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939817+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288409843+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253793877233e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716555371242e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1250703257977163+0j) [Y0 Z1 Y2] +
(-1.9332412771037803e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.002293956611352445+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123942+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714589251163e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771037803e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.002293956611352445+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123942+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714589251163e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231618+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781480387602e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986493198e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676601+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.551053917618788e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376507657042e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770581+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00553075921863151+0j) [Y0 Z1 Y2 Z4] +
(-1.38077814803876e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.376739308465975e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587162+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.38077814803876e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.376739308465975e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587162+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896507+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076823+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305986020256e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960922+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253794463274e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332103171313e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821319+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005354+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773244506028e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005354+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244506028e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666108+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132031+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674751001+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914647617e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.41829157464426e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504241+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599371486e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599371486e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018412067+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797223+0j) [Y0 Z1 Y2 Z12] +
(0.002326230623158027+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441859+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.5233896780272206e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178847+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199191065e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311866+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155211+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776288+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975374193e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660387+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464916172e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381029+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630346+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744706795e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624710785e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639197+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441859+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.5233896780272206e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178847+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199191065e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311866+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155211+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776288+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975374193e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660387+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464916172e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381029+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630346+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744706795e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624710785e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639197+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562548+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.202076878510505e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125413+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024423+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125413+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024423+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694864935834e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.684915095153227e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004446+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978540359136e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441896+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209154590184e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.09225061603563e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980136+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.09225061603563e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980136+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599610453856e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310131356678e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.0039898414566193014+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126902+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742645876e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823386+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823386+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453083077043e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363216887463e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536651729868e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209154590186e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.0001940085702975689+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538236+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289479097172e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446595688327e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369622+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696484+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265651575946e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209154590186e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.0001940085702975689+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538236+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479097172e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446595688327e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369622+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696484+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265651575946e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287494159068e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.042743277013781944+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487632+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.850564192804681e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487632+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192804681e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025528+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182525+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(1.0717282184183422e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.312094305281779e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839355+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425524599e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425524599e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803839+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914276+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907453+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328692+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303551523+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207106256e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422169207e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0032675138544235316+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328692+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303551523+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207106256e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422169207e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0032675138544235316+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336927+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413724025e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336927+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413724025e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002644+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002141361223101569+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046455+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245444+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219104+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219104+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009015301563e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476487301266e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658341897e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213160248e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.00153248352307306+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998843944724e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409987+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298118498e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278101+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.10551503724969e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226864+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.95607923005437e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001609531381721365+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221153446e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731754798062e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002462917007133906+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248909078+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.076732531910833e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.606071867706705e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496485+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389553708+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309316086426e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332622292967e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440404+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214168569+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402390533759e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651383+0j) [Y0 Y2] +
(3.117447945919852e-06+0j) [Y0 Z2 Z3 Y4] +
(0.045879470781297754+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861726+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453086924e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111777+0j) [Z0] +
(0.10433064780651383+0j) [Z0 X1 Z2 X3] +
(3.117447945919852e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.045879470781297754+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733861726+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453086924e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651383+0j) [Z0 Y1 Z2 Y3] +
(3.117447945919852e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.045879470781297754+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733861726+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453086924e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860495+0j) [Z0 Z1] +
(-8.337746753561754e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273128+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099213999+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735396642e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746753561754e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273128+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099213999+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735396642e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830437+0j) [Z0 Z2] +
(-1.1908508082499473e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329044+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.0763502195063498+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.580960369310894e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508082499473e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329044+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.0763502195063498+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.580960369310894e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.251294456745917+0j) [Z0 Z3] +
(-3.099349243575906e-06+0j) [Z0 X4 Z5 X6] +
(-1.531680879550431e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863617+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.099349243575906e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.531680879550431e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863617+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342153+0j) [Z0 Z4] +
(-3.3440815564490993e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585305923362e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0906514420703647+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815564490993e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585305923362e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0906514420703647+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360834+0j) [Z0 Z5] +
(0.05608468124661382+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209669494323e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661382+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209669494323e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017156+0j) [Z0 Z6] +
(0.056007330877807945+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.4818518339525095e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056007330877807945+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.4818518339525095e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314208+0j) [Z0 Z7] +
(0.27232518306605685+0j) [Z0 Z8] +
(0.2788345442672341+0j) [Z0 Z9] +
(-2.1776646050019886e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646050019886e-06+0j) [Z0 Y10 Z11 Y12] +
(0.1929972393536423+0j) [Z0 Z10] +
(-1.6148794138580225e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794138580225e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441757+0j) [Z0 Z11] +
(0.21102659849791472+0j) [Z0 Z12] +
(0.21631037498631767+0j) [Z0 Z13] +
(1.9332412771037803e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352445+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553123942+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714589251163e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441859+0j) [X1 X2 X4 X5] +
(-8.091637199191065e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311866+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.5233896780272206e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178847+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [X1 X2 X6 X7] +
(0.005114473831660387+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464916172e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776288+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975374193e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381029+0j) [X1 X2 X8 X9] +
(-0.0017992194936630346+0j) [X1 X2 X10 X11] +
(-5.287660624710785e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744706795e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639197+0j) [X1 X2 X12 X13] +
(-1.9332412771037803e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352445+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553123942+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714589251163e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441859+0j) [X1 Y2 Y4 X5] +
(-8.091637199191065e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311866+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5233896780272206e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178847+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [X1 Y2 Y6 X7] +
(0.005114473831660387+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464916172e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776288+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975374193e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381029+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630346+0j) [X1 Y2 Y10 X11] +
(-5.287660624710785e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744706795e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639197+0j) [X1 Y2 Y12 X13] +
(0.12507032579771632+0j) [X1 Z2 X3] +
(-1.38077814803876e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.376739308465975e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587162+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.38077814803876e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.376739308465975e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587162+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896507+0j) [X1 Z2 X3 Z4] +
(-1.551053917618788e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376507657042e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770581+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480387602e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986493198e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676601+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00553075921863151+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005354+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773244506028e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005354+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244506028e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666108+0j) [X1 Z2 X3 Z6] +
(0.005708495985960922+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332103171313e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253794463274e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076823+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305986020256e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821319+0j) [X1 Z2 X3 Z7] +
(0.002929768674751001+0j) [X1 Z2 X3 Z8] +
(0.011055020596132031+0j) [X1 Z2 X3 Z9] +
(-1.1076325599371486e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599371486e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018412067+0j) [X1 Z2 X3 Z10] +
(-6.41829157464426e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914647617e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504241+0j) [X1 Z2 X3 Z11] +
(0.002326230623158027+0j) [X1 Z2 X3 Z12] +
(0.006901238249797223+0j) [X1 Z2 X3 Z13] +
(-3.5682475211638447e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939817+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716555371242e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288409843+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253793877233e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125413+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024423+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209154590186e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538236+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0001940085702975689+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479097172e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446595688327e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696484+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369622+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565157594e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125413+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024423+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209154590186e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538236+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0001940085702975689+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479097172e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446595688327e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696484+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369622+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565157594e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768785105055e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.09225061603563e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980136+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.09225061603563e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980136+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978540359136e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441896+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.684915095153227e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004446+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209154590184e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310131356678e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.2362599610453856e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823386+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823386+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453083077043e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126902+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.0039898414566193014+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742645876e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536651729868e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363216887463e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562548+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0005192743499487632+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.850564192804681e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328692+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303551523+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422169207e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207106256e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423531+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487632+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.850564192804681e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328692+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303551523+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422169207e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207106256e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423531+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.042743277013781944+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.004636976661182525+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425524599e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425524599e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803839+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.312094305281779e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282184183422e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839355+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907453+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914276+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287494159068e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.0038764708993369278+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413724025e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.0038764708993369278+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413724025e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219113+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219113+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002652+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245444+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046455+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009015301563e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476487301267e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213160248e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.002141361223101569+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658341897e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409987+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298118498e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.00153248352307306+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998843944724e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226864+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.95607923005437e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025528+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278101+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.10551503724969e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.002462917007133906+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248909078+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.076732531910833e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694864935834e-07+0j) [X1 Z2 Z3 X5] +
(0.001609531381721365+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221153446e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731754798062e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332622292967e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440404+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214168569+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402390533759e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.003276971931231618+0j) [X1 X3] +
(3.606071867706705e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496485+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389553708+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309316086426e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771037803e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352445+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553123942+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714589251163e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441859+0j) [Y1 X2 X4 Y5] +
(-8.091637199191065e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311866+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.5233896780272206e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178847+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [Y1 X2 X6 Y7] +
(0.005114473831660387+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464916172e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776288+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975374193e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381029+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630346+0j) [Y1 X2 X10 Y11] +
(-5.287660624710785e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744706795e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639197+0j) [Y1 X2 X12 Y13] +
(1.9332412771037803e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352445+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553123942+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714589251163e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441859+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199191065e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311866+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.5233896780272206e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178847+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660387+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464916172e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776288+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975374193e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381029+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630346+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624710785e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744706795e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639197+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475211638447e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939817+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288409843+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253793877233e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716555371242e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579771632+0j) [Y1 Z2 Y3] +
(-1.38077814803876e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.376739308465975e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587162+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.38077814803876e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.376739308465975e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587162+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896507+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781480387602e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986493198e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676601+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.551053917618788e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376507657042e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770581+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00553075921863151+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005354+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773244506028e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005354+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244506028e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666108+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076823+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305986020256e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960922+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253794463274e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332103171313e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821319+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674751001+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132031+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599371486e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599371486e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018412067+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914647617e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.41829157464426e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504241+0j) [Y1 Z2 Y3 Z11] +
(0.002326230623158027+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797223+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125413+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024423+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209154590186e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538236+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0001940085702975689+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479097172e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446595688327e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696484+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369622+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565157594e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125413+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024423+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209154590186e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538236+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0001940085702975689+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479097172e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446595688327e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696484+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369622+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565157594e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562548+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2020768785105055e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.09225061603563e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980136+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.09225061603563e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980136+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.684915095153227e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004446+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978540359136e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441896+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209154590184e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310131356678e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.2362599610453856e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823386+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823386+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453083077043e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.0039898414566193014+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126902+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742645876e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536651729868e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363216887463e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487632+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.850564192804681e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328692+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303551523+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422169207e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207106256e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423531+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487632+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.850564192804681e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328692+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303551523+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422169207e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207106256e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423531+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287494159068e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.042743277013781944+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.004636976661182525+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425524599e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425524599e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803839+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282184183422e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.312094305281779e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839355+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907453+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914276+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.0038764708993369278+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413724025e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.0038764708993369278+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413724025e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219113+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219113+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002652+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245444+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046455+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009015301563e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476487301267e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213160248e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.002141361223101569+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658341897e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409987+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298118498e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.00153248352307306+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998843944724e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226864+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.95607923005437e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025528+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278101+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.10551503724969e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.002462917007133906+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248909078+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.076732531910833e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694864935834e-07+0j) [Y1 Z2 Z3 Y5] +
(0.001609531381721365+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221153446e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731754798062e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332622292967e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440404+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214168569+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402390533759e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231618+0j) [Y1 Y3] +
(3.606071867706705e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496485+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389553708+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309316086426e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111777+0j) [Z1] +
(-1.1908508082499473e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329044+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.0763502195063498+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.580960369310894e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508082499473e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329044+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.0763502195063498+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.580960369310894e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.251294456745917+0j) [Z1 Z2] +
(-8.337746753561754e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273128+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099213999+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735396642e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746753561754e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273128+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099213999+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735396642e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830437+0j) [Z1 Z3] +
(-3.3440815564490993e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585305923362e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0906514420703647+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815564490993e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585305923362e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0906514420703647+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360834+0j) [Z1 Z4] +
(-3.099349243575906e-06+0j) [Z1 X5 Z6 X7] +
(-1.531680879550431e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863617+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.099349243575906e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.531680879550431e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863617+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342153+0j) [Z1 Z5] +
(0.056007330877807945+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.4818518339525095e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056007330877807945+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.4818518339525095e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314208+0j) [Z1 Z6] +
(0.05608468124661382+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209669494323e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661382+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209669494323e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017156+0j) [Z1 Z7] +
(0.2788345442672341+0j) [Z1 Z8] +
(0.27232518306605685+0j) [Z1 Z9] +
(-1.6148794138580225e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794138580225e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441757+0j) [Z1 Z10] +
(-2.1776646050019886e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646050019886e-06+0j) [Z1 Y11 Z12 Y13] +
(0.1929972393536423+0j) [Z1 Z11] +
(0.21631037498631767+0j) [Z1 Z12] +
(0.21102659849791472+0j) [Z1 Z13] +
(-0.03583956795335341+0j) [X2 X3 Y4 Y5] +
(-2.1990516184562778e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.360956320326901e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01031148248983186+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618456278e-07+0j) [X2 X3 X5 X6] +
(-2.360956320326901e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.01031148248983186+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.03114381798896718+0j) [X2 X3 Y6 Y7] +
(0.005368659358109634+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350646838173e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109634+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350646838173e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904269+0j) [X2 X3 Y8 Y9] +
(-0.02538465750845733+0j) [X2 X3 Y10 Y11] +
(2.1726691015295404e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691015295404e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976429+0j) [X2 X3 Y12 Y13] +
(0.03583956795335341+0j) [X2 Y3 Y4 X5] +
(2.1990516184562778e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.360956320326901e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.01031148248983186+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618456278e-07+0j) [X2 Y3 Y5 X6] +
(-2.360956320326901e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.01031148248983186+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03114381798896718+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109634+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350646838173e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109634+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350646838173e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904269+0j) [X2 Y3 Y8 X9] +
(0.02538465750845733+0j) [X2 Y3 Y10 X11] +
(-2.1726691015295404e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691015295404e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976429+0j) [X2 Y3 Y12 X13] +
(-3.887051672960151e-06+0j) [X2 Z3 X4] +
(-0.005143391768825145+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962555+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706295152e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825145+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962555+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706295152e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411890384e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489514913093e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908957+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780962300116e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411218181e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343915499494e-07+0j) [X2 Z3 X4 Z6] +
(3.211842019123947e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.01929956057936377+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019123947e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.01929956057936377+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098728872e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423780092051e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052994488128e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380167+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221723+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564320171917e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068876+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068876+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.80170750055004e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541845575353e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306754485e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532435424148e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796775+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158446+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842449027882e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463111750522e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251586+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676326385e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454814+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372497332e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068532849e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847154+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688706+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122196949e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842449027882e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463111750522e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251586+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676326385e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454814+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372497332e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068532849e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847154+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688706+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122196949e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042247+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791024006+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381545812507e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791024006+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381545812507e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802101+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.00580518898982692+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646127+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288547132e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231084907595e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956451+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.745518400490522e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.745518400490522e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130914+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.000665007021949966+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.003493790359890058+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.56144718005637e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819206+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.01522563075722656+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507113395977e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.5443954293452347e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840024+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819206+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.01522563075722656+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507113395977e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.5443954293452347e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840024+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162063+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990714006824e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162063+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990714006824e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022755+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946562502463e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946562502463e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.02428211735469297+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314604+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898737+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158674+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158674+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505273093304e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765761147177e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327568871e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.8462016713186245e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205293+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793474669e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289087+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.1055267220739864e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600742+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501978184e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.02990378951262475+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656559434e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907867+0j) [X2 Z3 Z4 X6] +
(-0.01888903030494289+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.947356011767105e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0034795118903343573+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905446+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718062257e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167407313698e-06+0j) [X2 X4] +
(0.000495676231491544+0j) [X2 Z4 Z5 X6] +
(-0.035608378988312483+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348203195e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335341+0j) [Y2 X3 X4 Y5] +
(2.1990516184562778e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.360956320326901e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.01031148248983186+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618456278e-07+0j) [Y2 X3 X5 Y6] +
(-2.360956320326901e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.01031148248983186+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.03114381798896718+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109634+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350646838173e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109634+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350646838173e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904269+0j) [Y2 X3 X8 Y9] +
(0.02538465750845733+0j) [Y2 X3 X10 Y11] +
(-2.1726691015295404e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691015295404e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976429+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335341+0j) [Y2 Y3 X4 X5] +
(-2.1990516184562778e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.360956320326901e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01031148248983186+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618456278e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.360956320326901e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.01031148248983186+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.03114381798896718+0j) [Y2 Y3 X6 X7] +
(0.005368659358109634+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350646838173e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109634+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350646838173e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904269+0j) [Y2 Y3 X8 X9] +
(-0.02538465750845733+0j) [Y2 Y3 X10 X11] +
(2.1726691015295404e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691015295404e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976429+0j) [Y2 Y3 X12 X13] +
(1.6288532435424148e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796775+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158446+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051672960151e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825145+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962555+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706295152e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825145+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962555+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706295152e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411890384e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780962300116e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411218181e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489514913093e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908957+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343915499494e-07+0j) [Y2 Z3 Y4 Z6] +
(3.211842019123947e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.01929956057936377+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019123947e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.01929956057936377+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098728872e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423780092051e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052994488128e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221723+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380167+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564320171917e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068876+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068876+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.80170750055004e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541845575353e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306754485e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842449027882e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463111750522e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251586+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676326385e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454814+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372497332e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068532849e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847154+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688706+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122196949e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842449027882e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463111750522e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251586+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676326385e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454814+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372497332e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068532849e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847154+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688706+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122196949e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.56144718005637e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042247+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791024006+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381545812507e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791024006+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381545812507e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802101+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.00580518898982692+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646127+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231084907595e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288547132e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956451+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.745518400490522e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.745518400490522e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130914+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.000665007021949966+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.003493790359890058+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819206+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.01522563075722656+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507113395977e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.5443954293452347e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840024+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819206+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.01522563075722656+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507113395977e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.5443954293452347e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840024+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162063+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990714006824e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162063+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990714006824e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022755+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946562502463e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946562502463e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.02428211735469297+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314604+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898737+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158674+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158674+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505273093304e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765761147177e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327568871e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.8462016713186245e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205293+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793474669e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289087+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.1055267220739864e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600742+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501978184e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.02990378951262475+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656559434e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907867+0j) [Y2 Z3 Z4 Y6] +
(-0.01888903030494289+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.947356011767105e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0034795118903343573+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905446+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718062257e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167407313698e-06+0j) [Y2 Y4] +
(0.000495676231491544+0j) [Y2 Z4 Z5 Y6] +
(-0.035608378988312483+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348203195e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831697+0j) [Z2] +
(1.6021167407313698e-06+0j) [Z2 X3 Z4 X5] +
(0.000495676231491544+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.035608378988312483+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348203195e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167407313698e-06+0j) [Z2 Y3 Z4 Y5] +
(0.000495676231491544+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.035608378988312483+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348203195e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751358+0j) [Z2 Z3] +
(-9.509249751757619e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843147296599e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829995+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751757619e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147296599e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829995+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503217+0j) [Z2 Z4] +
(-1.17083013702139e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467623499e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.034903343373661855+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.17083013702139e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467623499e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.034903343373661855+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1607976453483856+0j) [Z2 Z5] +
(0.019020423173040056+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156046385026e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173040056+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156046385026e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1373910476268321+0j) [Z2 Z6] +
(0.02438908253114969+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.011122098170121e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.02438908253114969+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.011122098170121e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579928+0j) [Z2 Z7] +
(0.15071408121008292+0j) [Z2 Z8] +
(0.1869082047691256+0j) [Z2 Z9] +
(-1.063228342316914e-06+0j) [Z2 X10 Z11 X12] +
(-1.063228342316914e-06+0j) [Z2 Y10 Z11 Y12] +
(0.127995024924684+0j) [Z2 Z10] +
(1.1094407592126267e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407592126267e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314132+0j) [Z2 Z11] +
(0.14011289865354792+0j) [Z2 Z12] +
(0.15569010671752434+0j) [Z2 Z13] +
(0.005143391768825145+0j) [X3 X4 Y5 Y6] +
(0.009841749246962555+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706295152e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449027882e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676326385e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454814+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463111750522e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251586+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372497333e-07+0j) [X3 X4 X8 X9] +
(-4.643051068532849e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688706+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847154+0j) [X3 X4 Y11 Y12] +
(5.275883122196949e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825145+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962555+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706295152e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449027882e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676326385e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454814+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463111750522e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251586+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372497333e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068532849e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688706+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847154+0j) [X3 Y4 Y11 X12] +
(5.275883122196949e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051672960151e-06+0j) [X3 Z4 X5] +
(3.211842019123947e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.01929956057936377+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019123947e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.01929956057936377+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098728872e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489514913093e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908957+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780962300116e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411218181e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343915499494e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052994488128e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423780092051e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068876+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068876+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.80170750055004e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380167+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221723+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564320171917e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306754485e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541845575353e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532435424148e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796775+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158446+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791024006+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815458125077e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819206+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.01522563075722656+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.5443954293452347e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507113395977e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840024+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791024006+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815458125077e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819206+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.01522563075722656+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.5443954293452347e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507113395977e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840024+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042248+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646127+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.00580518898982692+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.745518400490522e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.745518400490522e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130914+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288547132e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231084907595e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956451+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.003493790359890058+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.000665007021949966+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.56144718005637e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162063+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990714006824e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162063+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990714006824e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946562502463e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.002446497155415868+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946562502463e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.002446497155415868+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.28164257767022743+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898737+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314604+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527309332e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765761147177e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.8462016713186245e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.02428211735469297+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327568871e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289087+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.1055267220739864e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205293+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793474669e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.02990378951262475+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656559434e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02599617759802101+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600742+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501978184e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0034795118903343573+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905446+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718062257e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994118903839e-07+0j) [X3 X5] +
(0.0016638798784907867+0j) [X3 Z5 Z6 X7] +
(-0.01888903030494289+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.947356011767105e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825145+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962555+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706295152e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449027882e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676326385e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454814+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463111750522e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251586+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372497333e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068532849e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688706+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847154+0j) [Y3 X4 X11 Y12] +
(5.275883122196949e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825145+0j) [Y3 Y4 X5 X6] +
(0.009841749246962555+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706295152e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449027882e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676326385e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454814+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463111750522e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251586+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372497333e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068532849e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688706+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847154+0j) [Y3 Y4 X11 X12] +
(5.275883122196949e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532435424148e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796775+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158446+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051672960151e-06+0j) [Y3 Z4 Y5] +
(3.211842019123947e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.01929956057936377+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019123947e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.01929956057936377+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098728872e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780962300116e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411218181e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489514913093e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908957+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343915499494e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052994488128e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423780092051e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068876+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068876+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.80170750055004e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221723+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380167+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564320171917e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306754485e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541845575353e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791024006+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815458125077e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819206+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.01522563075722656+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.5443954293452347e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507113395977e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840024+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791024006+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815458125077e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819206+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.01522563075722656+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.5443954293452347e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507113395977e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840024+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.56144718005637e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042248+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646127+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.00580518898982692+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.745518400490522e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.745518400490522e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130914+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231084907595e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288547132e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956451+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.003493790359890058+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.000665007021949966+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162063+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990714006824e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162063+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990714006824e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946562502463e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.002446497155415868+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946562502463e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.002446497155415868+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.28164257767022743+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898737+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314604+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527309332e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765761147177e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.8462016713186245e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.02428211735469297+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327568871e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289087+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.1055267220739864e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205293+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793474669e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.02990378951262475+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656559434e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802101+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600742+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501978184e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0034795118903343573+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905446+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718062257e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118903839e-07+0j) [Y3 Y5] +
(0.0016638798784907867+0j) [Y3 Z5 Z6 Y7] +
(-0.01888903030494289+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.947356011767105e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.653894222683169+0j) [Z3] +
(-1.17083013702139e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467623499e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.034903343373661855+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.17083013702139e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467623499e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.034903343373661855+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1607976453483856+0j) [Z3 Z4] +
(-9.509249751757619e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147296599e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829995+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751757619e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147296599e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829995+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503217+0j) [Z3 Z5] +
(0.02438908253114969+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.011122098170121e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.02438908253114969+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.011122098170121e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579928+0j) [Z3 Z6] +
(0.019020423173040056+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156046385026e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173040056+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156046385026e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1373910476268321+0j) [Z3 Z7] +
(0.1869082047691256+0j) [Z3 Z8] +
(0.15071408121008292+0j) [Z3 Z9] +
(1.1094407592126267e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407592126267e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314132+0j) [Z3 Z10] +
(-1.063228342316914e-06+0j) [Z3 X11 Z12 X13] +
(-1.063228342316914e-06+0j) [Z3 Y11 Z12 Y13] +
(0.127995024924684+0j) [Z3 Z11] +
(0.15569010671752434+0j) [Z3 Z12] +
(0.14011289865354792+0j) [Z3 Z13] +
(-0.011982389010247986+0j) [X4 X5 Y6 Y7] +
(-0.0073067599288329675+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293595185764e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0073067599288329675+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293595185764e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0071569349198569564+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248146+0j) [X4 X5 Y10 Y11] +
(-3.6945132945506096e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132945506096e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480382+0j) [X4 X5 Y12 Y13] +
(0.011982389010247986+0j) [X4 Y5 Y6 X7] +
(0.0073067599288329675+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293595185764e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0073067599288329675+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293595185764e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0071569349198569564+0j) [X4 Y5 Y8 X9] +
(0.01768006795248146+0j) [X4 Y5 Y10 X11] +
(3.6945132945506096e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132945506096e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480382+0j) [X4 Y5 Y12 X13] +
(-1.2260484988890058e-05+0j) [X4 Z5 X6] +
(-1.2283337824571228e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756956959+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824571228e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756956959+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579604296e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449081072181e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501831954452e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921582+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730651+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.692397828582337e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997614017+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997614017+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884881323e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155721449e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694649+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.83805275088227e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.97431171346366e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.01128519020084095+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535595+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.5565692182025676e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.83805275088227e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.97431171346366e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.01128519020084095+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535595+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.5565692182025676e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.330473188684188e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.0059237983365613405+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.330473188684188e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.0059237983365613405+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.63127792851686e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179517+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179517+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312894233715e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038820503e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775297347e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736476845e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736476845e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615611+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929528888+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.00961263460684727+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026786+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864702269e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638306+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.4443446760180815e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982172+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.2900284333152284e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289315+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.5183622157723515e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719745+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.92976581552766e-07+0j) [X4 X6] +
(-4.253224225607201e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601305+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247986+0j) [Y4 X5 X6 Y7] +
(0.0073067599288329675+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293595185764e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0073067599288329675+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293595185764e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0071569349198569564+0j) [Y4 X5 X8 Y9] +
(0.01768006795248146+0j) [Y4 X5 X10 Y11] +
(3.6945132945506096e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132945506096e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480382+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247986+0j) [Y4 Y5 X6 X7] +
(-0.0073067599288329675+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293595185764e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0073067599288329675+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293595185764e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0071569349198569564+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248146+0j) [Y4 Y5 X10 X11] +
(-3.6945132945506096e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132945506096e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480382+0j) [Y4 Y5 X12 X13] +
(0.008890731522694649+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484988890058e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824571228e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756956959+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824571228e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756956959+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579604296e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449081072181e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501831954452e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730651+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921582+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.692397828582337e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997614017+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997614017+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884881323e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155721449e-06+0j) [Y4 Z5 Y6 Z13] +
(4.83805275088227e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.97431171346366e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.01128519020084095+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535595+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.5565692182025676e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.83805275088227e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.97431171346366e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.01128519020084095+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535595+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.5565692182025676e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.330473188684188e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.0059237983365613405+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.330473188684188e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.0059237983365613405+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.63127792851686e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179517+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179517+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312894233715e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038820503e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775297347e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736476845e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736476845e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615611+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929528888+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.00961263460684727+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026786+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864702269e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638306+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.4443446760180815e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982172+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.2900284333152284e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289315+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.5183622157723515e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719745+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.92976581552766e-07+0j) [Y4 Y6] +
(-4.253224225607201e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601305+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145645+0j) [Z4] +
(-5.92976581552766e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225607201e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196013046+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.92976581552766e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225607201e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196013046+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985662+0j) [Z4 Z5] +
(0.018266834869375675+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174771049238e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375675+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174771049238e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040739+0j) [Z4 Z6] +
(0.01096007494054271+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468366235002e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01096007494054271+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468366235002e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1489943057506554+0j) [Z4 Z7] +
(0.15676396176430996+0j) [Z4 Z9] +
(1.8782101247747497e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247747497e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237598+0j) [Z4 Z10] +
(-1.8163031697758596e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031697758596e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485743+0j) [Z4 Z11] +
(0.11383573679388649+0j) [Z4 Z12] +
(0.15215040708869032+0j) [Z4 Z13] +
(1.228333782457123e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756956959+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.83805275088227e-07+0j) [X5 X6 X8 X9] +
(5.97431171346366e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535595+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.01128519020084095+0j) [X5 X6 Y11 Y12] +
(-4.556569218202566e-06+0j) [X5 X6 X12 X13] +
(-1.228333782457123e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756956959+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.83805275088227e-07+0j) [X5 Y6 Y8 X9] +
(5.97431171346366e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535595+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.01128519020084095+0j) [X5 Y6 Y11 X12] +
(-4.556569218202566e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988890061e-05+0j) [X5 Z6 X7] +
(-1.8818501831954452e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449081072181e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997614017+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997614017+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884881323e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921582+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730651+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.692397828582337e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155721449e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694649+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.330473188684188e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.0059237983365613405+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.330473188684188e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.0059237983365613405+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179517+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736476845e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179517+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736476845e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928516862e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775297347e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038820503e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615611+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02314513092952889+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026786+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.334331289423372e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.00961263460684727+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.4443446760180815e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982172+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864702269e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638306+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.5183622157723515e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719745+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608579604299e-06+0j) [X5 X7] +
(-6.2900284333152284e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289315+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782457123e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756956959+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.83805275088227e-07+0j) [Y5 X6 X8 Y9] +
(5.97431171346366e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535595+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.01128519020084095+0j) [Y5 X6 X11 Y12] +
(-4.556569218202566e-06+0j) [Y5 X6 X12 Y13] +
(1.228333782457123e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756956959+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.83805275088227e-07+0j) [Y5 Y6 Y8 Y9] +
(5.97431171346366e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535595+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.01128519020084095+0j) [Y5 Y6 X11 X12] +
(-4.556569218202566e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694649+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988890061e-05+0j) [Y5 Z6 Y7] +
(-1.8818501831954452e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449081072181e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997614017+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997614017+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884881323e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730651+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921582+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.692397828582337e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155721449e-06+0j) [Y5 Z6 Y7 Z12] +
(1.330473188684188e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.0059237983365613405+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.330473188684188e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.0059237983365613405+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179517+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736476845e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179517+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736476845e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928516862e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775297347e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038820503e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615611+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02314513092952889+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026786+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.334331289423372e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.00961263460684727+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.4443446760180815e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982172+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864702269e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638306+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.5183622157723515e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719745+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579604299e-06+0j) [Y5 Y7] +
(-6.2900284333152284e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289315+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145634+0j) [Z5] +
(0.01096007494054271+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468366235002e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01096007494054271+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468366235002e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1489943057506554+0j) [Z5 Z6] +
(0.018266834869375675+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174771049238e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375675+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174771049238e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040739+0j) [Z5 Z7] +
(0.15676396176430996+0j) [Z5 Z8] +
(-1.8163031697758596e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031697758596e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485743+0j) [Z5 Z10] +
(1.8782101247747497e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247747497e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237598+0j) [Z5 Z11] +
(0.15215040708869032+0j) [Z5 Z12] +
(0.11383573679388649+0j) [Z5 Z13] +
(-0.013873381748426037+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786623+0j) [X6 X7 Y10 Y11] +
(-1.0358477601132764e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477601132762e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651392+0j) [X6 X7 Y12 Y13] +
(0.013873381748426037+0j) [X6 Y7 Y8 X9] +
(0.017825140995786623+0j) [X6 Y7 Y10 X11] +
(1.0358477601132764e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477601132762e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651392+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110021+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393505677414e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110021+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393505677414e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564919027+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500133093e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500133093e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848258+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844582+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671545+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173038+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173038+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006999358e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559499181e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848532966e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228348399872e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.02981242451734595+0j) [X6 Z7 Z8 X10] +
(-3.2774831955095256e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456955+0j) [X6 Z7 Z9 X10] +
(-3.6102971305662995e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143988+0j) [X6 Z8 Z9 X10] +
(-3.769659451961907e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426037+0j) [Y6 X7 X8 Y9] +
(0.017825140995786623+0j) [Y6 X7 X10 Y11] +
(1.0358477601132764e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477601132762e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651392+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426037+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786623+0j) [Y6 Y7 X10 X11] +
(-1.0358477601132764e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477601132762e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651392+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110021+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393505677414e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110021+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393505677414e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564919027+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500133093e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500133093e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848258+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844582+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671545+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173038+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173038+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006999358e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559499181e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848532966e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228348399872e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.02981242451734595+0j) [Y6 Z7 Z8 Y10] +
(-3.2774831955095256e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456955+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971305662995e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143988+0j) [Y6 Z8 Z9 Y10] +
(-3.769659451961907e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615405+0j) [Z6] +
(0.030787505389143988+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.769659451961907e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143988+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.769659451961907e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270138+0j) [Z6 Z7] +
(0.1675665326546125+0j) [Z6 Z8] +
(0.18143991440303853+0j) [Z6 Z9] +
(-1.8551201215225836e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201215225836e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682655+0j) [Z6 Z10] +
(-2.89096788163586e-06+0j) [Z6 X11 Z12 X13] +
(-2.89096788163586e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261318+0j) [Z6 Z11] +
(0.1340171526196367+0j) [Z6 Z12] +
(0.1513832716142881+0j) [Z6 Z13] +
(-0.0002921986261110021+0j) [X7 X8 Y9 Y10] +
(3.3281393505677414e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110021+0j) [X7 Y8 Y9 X10] +
(-3.3281393505677414e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131455001330934e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173038+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131455001330934e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173038+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564919027+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671545+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844582+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086006999356e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559499182e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228348399872e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848258+0j) [X7 Z8 Z9 X11] +
(-6.524373848532966e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456955+0j) [X7 Z8 Z10 X11] +
(-3.6102971305662995e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.02981242451734595+0j) [X7 Z9 Z10 X11] +
(-3.2774831955095256e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110021+0j) [Y7 X8 X9 Y10] +
(-3.3281393505677414e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110021+0j) [Y7 Y8 X9 X10] +
(3.3281393505677414e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131455001330934e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173038+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131455001330934e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173038+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564919027+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671545+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844582+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086006999356e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559499182e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228348399872e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848258+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848532966e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456955+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971305662995e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.02981242451734595+0j) [Y7 Z9 Z10 Y11] +
(-3.2774831955095256e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615405+0j) [Z7] +
(0.18143991440303853+0j) [Z7 Z8] +
(0.1675665326546125+0j) [Z7 Z9] +
(-2.89096788163586e-06+0j) [Z7 X10 Z11 X12] +
(-2.89096788163586e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261318+0j) [Z7 Z10] +
(-1.8551201215225836e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201215225836e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682655+0j) [Z7 Z11] +
(0.1513832716142881+0j) [Z7 Z12] +
(0.1340171526196367+0j) [Z7 Z13] +
(-0.009560705729135909+0j) [X8 X9 Y10 Y11] +
(6.628614201680148e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201680148e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561838+0j) [X8 X9 Y12 Y13] +
(0.009560705729135909+0j) [X8 Y9 Y10 X11] +
(-6.628614201680148e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201680148e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561838+0j) [X8 Y9 Y12 X13] +
(0.009560705729135909+0j) [Y8 X9 X10 Y11] +
(-6.628614201680148e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201680148e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561838+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135909+0j) [Y8 Y9 X10 X11] +
(6.628614201680148e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201680148e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561838+0j) [Y8 Y9 X12 X13] +
(1.3693525634718182+0j) [Z8] +
(-1.5973171977964784e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171977964784e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852574+0j) [Z8 Z10] +
(-9.344557776284634e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557776284634e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766166+0j) [Z8 Z11] +
(0.14973486803496905+0j) [Z8 Z12] +
(0.15582269051553088+0j) [Z8 Z13] +
(1.3693525634718187+0j) [Z9] +
(-9.344557776284634e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557776284634e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766166+0j) [Z9 Z10] +
(-1.5973171977964784e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171977964784e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852574+0j) [Z9 Z11] +
(0.15582269051553088+0j) [Z9 Z12] +
(0.14973486803496905+0j) [Z9 Z13] +
(-0.028685183716105893+0j) [X10 X11 Y12 Y13] +
(0.028685183716105893+0j) [X10 Y11 Y12 X13] +
(-1.0722312157921233e-05+0j) [X10 Z11 X12] +
(7.9544131763531e-06+0j) [X10 Z11 X12 Z13] +
(-8.19426137221429e-06+0j) [X10 X12] +
(0.028685183716105893+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105893+0j) [Y10 Y11 X12 X13] +
(-1.0722312157921233e-05+0j) [Y10 Z11 Y12] +
(7.9544131763531e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.19426137221429e-06+0j) [Y10 Y12] +
(0.7829661725950209+0j) [Z10] +
(-8.19426137221429e-06+0j) [Z10 X11 Z12 X13] +
(-8.19426137221429e-06+0j) [Z10 Y11 Z12 Y13] +
(0.1492635514738888+0j) [Z10 Z11] +
(0.11270386920332204+0j) [Z10 Z12] +
(0.14138905291942794+0j) [Z10 Z13] +
(-1.0722312157921231e-05+0j) [X11 Z12 X13] +
(7.9544131763531e-06+0j) [X11 X13] +
(-1.0722312157921231e-05+0j) [Y11 Z12 Y13] +
(7.9544131763531e-06+0j) [Y11 Y13] +
(0.782966172595021+0j) [Z11] +
(0.14138905291942794+0j) [Z11 Z12] +
(0.11270386920332204+0j) [Z11 Z13] +
(0.8084581961720466+0j) [Z12] +
(0.154357486572236+0j) [Z12 Z13] +
(0.8084581961720468+0j) [Z13]
  (-46.463906788688895) [I0]
+ (0.7829661725950176) [Z10]
+ (0.7829661725950179) [Z11]
+ (0.8084581961720475) [Z12]
+ (0.8084581961720477) [Z13]
+ (1.2034402289145625) [Z5]
+ (1.203440228914563) [Z4]
+ (1.309686298861542) [Z7]
+ (1.3096862988615423) [Z6]
+ (1.3693525634718182) [Z8]
+ (1.3693525634718184) [Z9]
+ (1.6538942226831734) [Z3]
+ (1.6538942226831737) [Z2]
+ (-8.19426137183238e-06) [Y10 Y12]
+ (-8.19426137183238e-06) [X10 X12]
+ (-1.8540608578769836e-06) [Y5 Y7]
+ (-1.8540608578769836e-06) [X5 X7]
+ (-7.764994118103289e-07) [Y3 Y5]
+ (-7.764994118103289e-07) [X3 X5]
+ (-5.929765815327643e-07) [Y4 Y6]
+ (-5.929765815327643e-07) [X4 X6]
+ (1.6021167406521803e-06) [Y2 Y4]
+ (1.6021167406521803e-06) [X2 X4]
+ (7.954413175852768e-06) [Y11 Y13]
+ (7.954413175852768e-06) [X11 X13]
+ (0.0032769719312317146) [Y1 Y3]
+ (0.0032769719312317146) [X1 X3]
+ (0.10433064780651433) [Y0 Y2]
+ (0.10433064780651433) [X0 X2]
+ (0.1127038692033221) [Z10 Z12]
+ (0.1127038692033221) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.11952438964682655) [Z6 Z10]
+ (0.11952438964682655) [Z7 Z11]
+ (0.12489990917237587) [Z4 Z10]
+ (0.12489990917237587) [Z5 Z11]
+ (0.12495807739503234) [Z2 Z4]
+ (0.12495807739503234) [Z3 Z5]
+ (0.12799502492468418) [Z2 Z10]
+ (0.12799502492468418) [Z3 Z11]
+ (0.13401715261963698) [Z6 Z12]
+ (0.13401715261963698) [Z7 Z13]
+ (0.1370119167404074) [Z4 Z6]
+ (0.1370119167404074) [Z5 Z7]
+ (0.13734953064261313) [Z6 Z11]
+ (0.13734953064261313) [Z7 Z10]
+ (0.13739104762683238) [Z2 Z6]
+ (0.13739104762683238) [Z3 Z7]
+ (0.13766872645852568) [Z8 Z10]
+ (0.13766872645852568) [Z9 Z11]
+ (0.14011289865354828) [Z2 Z12]
+ (0.14011289865354828) [Z3 Z13]
+ (0.141389052919428) [Z10 Z13]
+ (0.141389052919428) [Z11 Z12]
+ (0.14257997712485743) [Z4 Z11]
+ (0.14257997712485743) [Z5 Z10]
+ (0.1472294321876616) [Z8 Z11]
+ (0.1472294321876616) [Z9 Z10]
+ (0.14899430575065536) [Z4 Z7]
+ (0.14899430575065536) [Z5 Z6]
+ (0.1492635514738888) [Z10 Z11]
+ (0.14960702684445293) [Z4 Z8]
+ (0.14960702684445293) [Z5 Z9]
+ (0.14973486803496927) [Z8 Z12]
+ (0.14973486803496927) [Z9 Z13]
+ (0.15071408121008306) [Z2 Z8]
+ (0.15071408121008306) [Z3 Z9]
+ (0.15138327161428838) [Z6 Z13]
+ (0.15138327161428838) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314148) [Z2 Z11]
+ (0.15337968243314148) [Z3 Z10]
+ (0.15435748657223633) [Z12 Z13]
+ (0.15569010671752476) [Z2 Z13]
+ (0.15569010671752476) [Z3 Z12]
+ (0.1558226905155311) [Z8 Z13]
+ (0.1558226905155311) [Z9 Z12]
+ (0.15676396176430987) [Z4 Z9]
+ (0.15676396176430987) [Z5 Z8]
+ (0.1575531479798566) [Z4 Z5]
+ (0.16079764534838575) [Z2 Z5]
+ (0.16079764534838575) [Z3 Z4]
+ (0.1685348656157996) [Z2 Z7]
+ (0.1685348656157996) [Z3 Z6]
+ (0.1814399144030386) [Z6 Z9]
+ (0.1814399144030386) [Z7 Z8]
+ (0.1818908579075141) [Z2 Z3]
+ (0.1869082047691258) [Z2 Z9]
+ (0.1869082047691258) [Z3 Z8]
+ (0.19299723935364213) [Z0 Z10]
+ (0.19299723935364213) [Z1 Z11]
+ (0.19392534613270163) [Z6 Z7]
+ (0.19661770890342128) [Z0 Z4]
+ (0.19661770890342128) [Z1 Z5]
+ (0.1993635453736081) [Z0 Z5]
+ (0.1993635453736081) [Z1 Z4]
+ (0.20072866460441735) [Z0 Z11]
+ (0.20072866460441735) [Z1 Z10]
+ (0.2110265984979151) [Z0 Z12]
+ (0.2110265984979151) [Z1 Z13]
+ (0.21631037498631805) [Z0 Z13]
+ (0.21631037498631805) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830462) [Z0 Z2]
+ (0.23671080783830462) [Z1 Z3]
+ (0.2416466393601717) [Z0 Z6]
+ (0.2416466393601717) [Z1 Z7]
+ (0.24853483371314225) [Z0 Z7]
+ (0.24853483371314225) [Z1 Z6]
+ (0.2512944567459174) [Z0 Z3]
+ (0.2512944567459174) [Z1 Z2]
+ (0.2723251830660567) [Z0 Z8]
+ (0.2723251830660567) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860487) [Z0 Z1]
+ (-1.2260484988626077e-05) [Y4 Z5 Y6]
+ (-1.2260484988626077e-05) [X4 Z5 X6]
+ (-1.2260484988626073e-05) [Y5 Z6 Y7]
+ (-1.2260484988626073e-05) [X5 Z6 X7]
+ (-1.0722312158290207e-05) [Y11 Z12 Y13]
+ (-1.0722312158290207e-05) [X11 Z12 X13]
+ (-1.0722312158290205e-05) [Y10 Z11 Y12]
+ (-1.0722312158290205e-05) [X10 Z11 X12]
+ (-3.8870516717107845e-06) [Y2 Z3 Y4]
+ (-3.8870516717107845e-06) [X2 Z3 X4]
+ (-3.8870516717107845e-06) [Y3 Z4 Y5]
+ (-3.8870516717107845e-06) [X3 Z4 X5]
+ (0.12507032579772212) [Y0 Z1 Y2]
+ (0.12507032579772212) [X0 Z1 X2]
+ (0.12507032579772215) [Y1 Z2 Y3]
+ (0.12507032579772215) [X1 Z2 X3]
+ (-0.038314670294803864) [Y4 Y5 X12 X13]
+ (-0.038314670294803864) [X4 X5 Y12 Y13]
+ (-0.03619412355904273) [Y2 Y3 X8 X9]
+ (-0.03619412355904273) [X2 X3 Y8 Y9]
+ (-0.03583956795335342) [Y2 Y3 X4 X5]
+ (-0.03583956795335342) [X2 X3 Y4 Y5]
+ (-0.031143817988967207) [Y2 Y3 X6 X7]
+ (-0.031143817988967207) [X2 X3 Y6 Y7]
+ (-0.0286851837161059) [Y10 Y11 X12 X13]
+ (-0.0286851837161059) [X10 X11 Y12 Y13]
+ (-0.025996177598021093) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021093) [X3 Z4 Z5 X7]
+ (-0.025384657508457316) [Y2 Y3 X10 X11]
+ (-0.025384657508457316) [X2 X3 Y10 Y11]
+ (-0.019028242443847192) [Y3 Y4 X11 X12]
+ (-0.019028242443847192) [X3 X4 Y11 Y12]
+ (-0.017825140995786578) [Y6 Y7 X10 X11]
+ (-0.017825140995786578) [X6 X7 Y10 Y11]
+ (-0.017680067952481556) [Y4 Y5 X10 X11]
+ (-0.017680067952481556) [X4 X5 Y10 Y11]
+ (-0.017366118994651375) [Y6 Y7 X12 X13]
+ (-0.017366118994651375) [X6 X7 Y12 Y13]
+ (-0.015577208063976474) [Y2 Y3 X12 X13]
+ (-0.015577208063976474) [X2 X3 Y12 Y13]
+ (-0.014583648907612743) [Y0 Y1 X2 X3]
+ (-0.014583648907612743) [X0 X1 Y2 Y3]
+ (-0.013873381748426051) [Y6 Y7 X8 X9]
+ (-0.013873381748426051) [X6 X7 Y8 Y9]
+ (-0.011982389010247967) [Y4 Y5 X6 X7]
+ (-0.011982389010247967) [X4 X5 Y6 Y7]
+ (-0.011285190200840957) [Y5 X6 X11 Y12]
+ (-0.011285190200840957) [X5 Y6 Y11 X12]
+ (-0.009560705729135895) [Y8 Y9 X10 X11]
+ (-0.009560705729135895) [X8 X9 Y10 Y11]
+ (-0.008125251921381041) [Y1 X2 X8 Y9]
+ (-0.008125251921381041) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381041) [X1 X2 X8 X9]
+ (-0.008125251921381041) [X1 Y2 Y8 X9]
+ (-0.00773142525077526) [Y0 Y1 X10 X11]
+ (-0.00773142525077526) [X0 X1 Y10 Y11]
+ (-0.007156934919856939) [Y4 Y5 X8 X9]
+ (-0.007156934919856939) [X4 X5 Y8 Y9]
+ (-0.006509361201177232) [Y0 Y1 X8 X9]
+ (-0.006509361201177232) [X0 X1 Y8 Y9]
+ (-0.00608782248056186) [Y8 Y9 X12 X13]
+ (-0.00608782248056186) [X8 X9 Y12 Y13]
+ (-0.005283776488402955) [Y0 Y1 X12 X13]
+ (-0.005283776488402955) [X0 X1 Y12 Y13]
+ (-0.0051433917688251535) [Y3 X4 X5 Y6]
+ (-0.0051433917688251535) [X3 Y4 Y5 X6]
+ (-0.00468490338815522) [Y1 X2 X6 Y7]
+ (-0.00468490338815522) [Y1 Y2 Y6 Y7]
+ (-0.00468490338815522) [X1 X2 X6 X7]
+ (-0.00468490338815522) [X1 Y2 Y6 X7]
+ (-0.004575007626639216) [Y1 X2 X12 Y13]
+ (-0.004575007626639216) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639216) [X1 X2 X12 X13]
+ (-0.004575007626639216) [X1 Y2 Y12 X13]
+ (-0.004424855449441854) [Y1 X2 X4 Y5]
+ (-0.004424855449441854) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441854) [X1 X2 X4 X5]
+ (-0.004424855449441854) [X1 Y2 Y4 X5]
+ (-0.0034795118903343625) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343625) [X2 Z3 Z5 X6]
+ (-0.0034795118903343625) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343625) [X3 Z4 Z6 X7]
+ (-0.002745836470186806) [Y0 Y1 X4 X5]
+ (-0.002745836470186806) [X0 X1 Y4 Y5]
+ (-0.0017992194936630216) [Y1 X2 X10 Y11]
+ (-0.0017992194936630216) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630216) [X1 X2 X10 X11]
+ (-0.0017992194936630216) [X1 Y2 Y10 X11]
+ (-0.0002921986261110074) [Y7 Y8 X9 X10]
+ (-0.0002921986261110074) [X7 X8 Y9 Y10]
+ (-8.19426137183238e-06) [Z10 Y11 Z12 Y13]
+ (-8.19426137183238e-06) [Z10 X11 Z12 X13]
+ (-7.801707500101582e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500101582e-06) [X2 Z3 X4 Z11]
+ (-7.801707500101582e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500101582e-06) [X3 Z4 X5 Z10]
+ (-4.643051068313862e-06) [Y3 X4 X10 Y11]
+ (-4.643051068313862e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068313862e-06) [X3 X4 X10 X11]
+ (-4.643051068313862e-06) [X3 Y4 Y10 X11]
+ (-4.588855155510016e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155510016e-06) [X4 Z5 X6 Z13]
+ (-4.588855155510016e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155510016e-06) [X5 Z6 X7 Z12]
+ (-4.556569217988823e-06) [Y5 X6 X12 Y13]
+ (-4.556569217988823e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217988823e-06) [X5 X6 X12 X13]
+ (-4.556569217988823e-06) [X5 Y6 Y12 X13]
+ (-3.6945132944260314e-06) [Y4 X5 X11 Y12]
+ (-3.6945132944260314e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132944260314e-06) [X4 X5 X11 X12]
+ (-3.6945132944260314e-06) [X4 Y5 Y11 X12]
+ (-3.344081556303551e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556303551e-06) [Z0 X5 Z6 X7]
+ (-3.344081556303551e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556303551e-06) [Z1 X4 Z5 X6]
+ (-3.1586564317877203e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564317877203e-06) [X2 Z3 X4 Z10]
+ (-3.1586564317877203e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564317877203e-06) [X3 Z4 X5 Z11]
+ (-3.0993492434440025e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492434440025e-06) [Z0 X4 Z5 X6]
+ (-3.0993492434440025e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492434440025e-06) [Z1 X5 Z6 X7]
+ (-2.8909678815458422e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678815458422e-06) [Z6 X11 Z12 X13]
+ (-2.8909678815458422e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678815458422e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050699782e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050699782e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050699782e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050699782e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831211958e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831211958e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831211958e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831211958e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215216597e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215216597e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215216597e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215216597e-06) [Z7 X11 Z12 X13]
+ (-1.8540608578769836e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608578769836e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697744488e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697744488e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697744488e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697744488e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285545364e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285545364e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285545364e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285545364e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139737192e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139737192e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139737192e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139737192e-06) [Z1 X10 Z11 X12]
+ (-1.5973171978057134e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171978057134e-06) [Z8 X10 Z11 X12]
+ (-1.5973171978057134e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171978057134e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489099767e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489099767e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489099767e-06) [X3 X4 X6 X7]
+ (-1.4548424489099767e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080717865e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080717865e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080717865e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080717865e-06) [X5 Z6 X7 Z9]
+ (-1.1954890097387898e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890097387898e-06) [X2 Z3 X4 Z7]
+ (-1.1954890097387898e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890097387898e-06) [X3 Z4 X5 Z6]
+ (-1.190850808028587e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808028587e-06) [Z0 X3 Z4 X5]
+ (-1.190850808028587e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808028587e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369842835e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369842835e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369842835e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369842835e-06) [Z3 X4 Z5 X6]
+ (-1.063228342343266e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342343266e-06) [Z2 X10 Z11 X12]
+ (-1.063228342343266e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342343266e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600241823e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600241823e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600241823e-06) [X6 X7 X11 X12]
+ (-1.0358477600241823e-06) [X6 Y7 Y11 X12]
+ (-9.509249751439655e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751439655e-07) [Z2 X4 Z5 X6]
+ (-9.509249751439655e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751439655e-07) [Z3 X5 Z6 X7]
+ (-9.344557776915864e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776915864e-07) [Z8 X11 Z12 X13]
+ (-9.344557776915864e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776915864e-07) [Z9 X10 Z11 X12]
+ (-8.33774675159929e-07) [Z0 Y2 Z3 Y4]
+ (-8.33774675159929e-07) [Z0 X2 Z3 X4]
+ (-8.33774675159929e-07) [Z1 Y3 Z4 Y5]
+ (-8.33774675159929e-07) [Z1 X3 Z4 X5]
+ (-7.956895371575605e-07) [Y3 X4 X8 Y9]
+ (-7.956895371575605e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371575605e-07) [X3 X4 X8 X9]
+ (-7.956895371575605e-07) [X3 Y4 Y8 X9]
+ (-7.764994118103289e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118103289e-07) [X2 Z3 X4 Z5]
+ (-5.929765815327643e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815327643e-07) [Z4 X5 Z6 X7]
+ (-5.770052993286443e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052993286443e-07) [X2 Z3 X4 Z9]
+ (-5.770052993286443e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052993286443e-07) [X3 Z4 X5 Z8]
+ (-5.471647744428599e-07) [Y1 Y2 X11 X12]
+ (-5.471647744428599e-07) [X1 X2 Y11 Y12]
+ (-4.838052750494092e-07) [Y5 X6 X8 Y9]
+ (-4.838052750494092e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750494092e-07) [X5 X6 X8 X9]
+ (-4.838052750494092e-07) [X5 Y6 Y8 X9]
+ (-3.570761328686578e-07) [Y0 X1 X3 Y4]
+ (-3.570761328686578e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328686578e-07) [X0 X1 X3 X4]
+ (-3.570761328686578e-07) [X0 Y1 Y3 X4]
+ (-2.4473231285954846e-07) [Y0 X1 X5 Y6]
+ (-2.4473231285954846e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231285954846e-07) [X0 X1 X5 X6]
+ (-2.4473231285954846e-07) [X0 Y1 Y5 X6]
+ (-2.1990516184031787e-07) [Y2 X3 X5 Y6]
+ (-2.1990516184031787e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516184031787e-07) [X2 X3 X5 X6]
+ (-2.1990516184031787e-07) [X2 Y3 Y5 X6]
+ (-1.9332412769306473e-07) [Y1 X2 X3 Y4]
+ (-1.9332412769306473e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861053755e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861053755e-07) [X1 Z2 Z3 X5]
+ (1.7379332623288818e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623288818e-07) [X0 Z1 Z3 X4]
+ (1.7379332623288818e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623288818e-07) [X1 Z2 Z4 X5]
+ (1.9332412769306473e-07) [Y1 Y2 X3 X4]
+ (1.9332412769306473e-07) [X1 X2 Y3 Y4]
+ (2.1868423782891625e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423782891625e-07) [X2 Z3 X4 Z8]
+ (2.1868423782891625e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423782891625e-07) [X3 Z4 X5 Z9]
+ (2.5935343917118693e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343917118693e-07) [X2 Z3 X4 Z6]
+ (2.5935343917118693e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343917118693e-07) [X3 Z4 X5 Z7]
+ (3.606071867725521e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867725521e-07) [X0 Z1 Z2 X4]
+ (3.606071867725521e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867725521e-07) [X1 Z3 Z4 X5]
+ (5.471647744428599e-07) [Y1 X2 X11 Y12]
+ (5.471647744428599e-07) [X1 Y2 Y11 X12]
+ (5.627851910962587e-07) [Y0 X1 X11 Y12]
+ (5.627851910962587e-07) [Y0 Y1 Y11 Y12]
+ (5.627851910962587e-07) [X0 X1 X11 X12]
+ (5.627851910962587e-07) [X0 Y1 Y11 X12]
+ (6.62861420114127e-07) [Y8 X9 X11 Y12]
+ (6.62861420114127e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420114127e-07) [X8 X9 X11 X12]
+ (6.62861420114127e-07) [X8 Y9 Y11 X12]
+ (1.1094407590020529e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590020529e-06) [Z2 X11 Z12 X13]
+ (1.1094407590020529e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590020529e-06) [Z3 X10 Z11 X12]
+ (1.6021167406521803e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406521803e-06) [Z2 X3 Z4 X5]
+ (1.878210124651583e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124651583e-06) [Z4 X10 Z11 X12]
+ (1.878210124651583e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124651583e-06) [Z5 X11 Z12 X13]
+ (2.172669101345319e-06) [Y2 X3 X11 Y12]
+ (2.172669101345319e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101345319e-06) [X2 X3 X11 X12]
+ (2.172669101345319e-06) [X2 Y3 Y11 X12]
+ (3.1174479457665256e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479457665256e-06) [X0 Z2 Z3 X4]
+ (3.5390541844212332e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844212332e-06) [X2 Z3 X4 Z12]
+ (3.5390541844212332e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844212332e-06) [X3 Z4 X5 Z13]
+ (4.2819138845924525e-06) [Y4 Z5 Y6 Z11]
+ (4.2819138845924525e-06) [X4 Z5 X6 Z11]
+ (4.2819138845924525e-06) [Y5 Z6 Y7 Z10]
+ (4.2819138845924525e-06) [X5 Z6 X7 Z10]
+ (5.2758831218647154e-06) [Y3 X4 X12 Y13]
+ (5.2758831218647154e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831218647154e-06) [X3 X4 X12 X13]
+ (5.2758831218647154e-06) [X3 Y4 Y12 X13]
+ (5.97431171314699e-06) [Y5 X6 X10 Y11]
+ (5.97431171314699e-06) [Y5 Y6 Y10 Y11]
+ (5.97431171314699e-06) [X5 X6 X10 X11]
+ (5.97431171314699e-06) [X5 Y6 Y10 X11]
+ (7.954413175852768e-06) [Y10 Z11 Y12 Z13]
+ (7.954413175852768e-06) [X10 Z11 X12 Z13]
+ (8.814937306285949e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306285949e-06) [X2 Z3 X4 Z13]
+ (8.814937306285949e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306285949e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110074) [Y7 X8 X9 Y10]
+ (0.0002921986261110074) [X7 Y8 Y9 X10]
+ (0.0004956762314915417) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915417) [X2 Z4 Z5 X6]
+ (0.0011059037691897107) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897107) [X0 Z1 X2 Z5]
+ (0.0011059037691897107) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897107) [X1 Z2 X3 Z4]
+ (0.001663879878490792) [Y2 Z3 Z4 Y6]
+ (0.001663879878490792) [X2 Z3 Z4 X6]
+ (0.001663879878490792) [Y3 Z5 Z6 Y7]
+ (0.001663879878490792) [X3 Z5 Z6 X7]
+ (0.001756070701841264) [Y0 Z1 Y2 Z11]
+ (0.001756070701841264) [X0 Z1 X2 Z11]
+ (0.001756070701841264) [Y1 Z2 Y3 Z10]
+ (0.001756070701841264) [X1 Z2 X3 Z10]
+ (0.002326230623158113) [Y0 Z1 Y2 Z13]
+ (0.002326230623158113) [X0 Z1 X2 Z13]
+ (0.002326230623158113) [Y1 Z2 Y3 Z12]
+ (0.002326230623158113) [X1 Z2 X3 Z12]
+ (0.002745836470186806) [Y0 X1 X4 Y5]
+ (0.002745836470186806) [X0 Y1 Y4 X5]
+ (0.002929768674751086) [Y0 Z1 Y2 Z9]
+ (0.002929768674751086) [X0 Z1 X2 Z9]
+ (0.002929768674751086) [Y1 Z2 Y3 Z8]
+ (0.002929768674751086) [X1 Z2 X3 Z8]
+ (0.003276971931231714) [Y0 Z1 Y2 Z3]
+ (0.003276971931231714) [X0 Z1 X2 Z3]
+ (0.0033476175306662013) [Y0 Z1 Y2 Z7]
+ (0.0033476175306662013) [X0 Z1 X2 Z7]
+ (0.0033476175306662013) [Y1 Z2 Y3 Z6]
+ (0.0033476175306662013) [X1 Z2 X3 Z6]
+ (0.003555290195504286) [Y0 Z1 Y2 Z10]
+ (0.003555290195504286) [X0 Z1 X2 Z10]
+ (0.003555290195504286) [Y1 Z2 Y3 Z11]
+ (0.003555290195504286) [X1 Z2 X3 Z11]
+ (0.0051433917688251535) [Y3 Y4 X5 X6]
+ (0.0051433917688251535) [X3 X4 Y5 Y6]
+ (0.005283776488402955) [Y0 X1 X12 Y13]
+ (0.005283776488402955) [X0 Y1 Y12 X13]
+ (0.005530759218631565) [Y0 Z1 Y2 Z4]
+ (0.005530759218631565) [X0 Z1 X2 Z4]
+ (0.005530759218631565) [Y1 Z2 Y3 Z5]
+ (0.005530759218631565) [X1 Z2 X3 Z5]
+ (0.00608782248056186) [Y8 X9 X12 Y13]
+ (0.00608782248056186) [X8 Y9 Y12 X13]
+ (0.006509361201177232) [Y0 X1 X8 Y9]
+ (0.006509361201177232) [X0 Y1 Y8 X9]
+ (0.006901238249797328) [Y0 Z1 Y2 Z12]
+ (0.006901238249797328) [X0 Z1 X2 Z12]
+ (0.006901238249797328) [Y1 Z2 Y3 Z13]
+ (0.006901238249797328) [X1 Z2 X3 Z13]
+ (0.007156934919856939) [Y4 X5 X8 Y9]
+ (0.007156934919856939) [X4 Y5 Y8 X9]
+ (0.00773142525077526) [Y0 X1 X10 Y11]
+ (0.00773142525077526) [X0 Y1 Y10 X11]
+ (0.008032520918821421) [Y0 Z1 Y2 Z6]
+ (0.008032520918821421) [X0 Z1 X2 Z6]
+ (0.008032520918821421) [Y1 Z2 Y3 Z7]
+ (0.008032520918821421) [X1 Z2 X3 Z7]
+ (0.009560705729135895) [Y8 X9 X10 Y11]
+ (0.009560705729135895) [X8 Y9 Y10 X11]
+ (0.011055020596132127) [Y0 Z1 Y2 Z8]
+ (0.011055020596132127) [X0 Z1 X2 Z8]
+ (0.011055020596132127) [Y1 Z2 Y3 Z9]
+ (0.011055020596132127) [X1 Z2 X3 Z9]
+ (0.011285190200840957) [Y5 Y6 X11 X12]
+ (0.011285190200840957) [X5 X6 Y11 Y12]
+ (0.011307274008848185) [Y7 Z8 Z9 Y11]
+ (0.011307274008848185) [X7 Z8 Z9 X11]
+ (0.011982389010247967) [Y4 X5 X6 Y7]
+ (0.011982389010247967) [X4 Y5 Y6 X7]
+ (0.013873381748426051) [Y6 X7 X8 Y9]
+ (0.013873381748426051) [X6 Y7 Y8 X9]
+ (0.014583648907612743) [Y0 X1 X2 Y3]
+ (0.014583648907612743) [X0 Y1 Y2 X3]
+ (0.015577208063976474) [Y2 X3 X12 Y13]
+ (0.015577208063976474) [X2 Y3 Y12 X13]
+ (0.017366118994651375) [Y6 X7 X12 Y13]
+ (0.017366118994651375) [X6 Y7 Y12 X13]
+ (0.017680067952481556) [Y4 X5 X10 Y11]
+ (0.017680067952481556) [X4 Y5 Y10 X11]
+ (0.017825140995786578) [Y6 X7 X10 Y11]
+ (0.017825140995786578) [X6 Y7 Y10 X11]
+ (0.019028242443847192) [Y3 X4 X11 Y12]
+ (0.019028242443847192) [X3 Y4 Y11 X12]
+ (0.025384657508457316) [Y2 X3 X10 Y11]
+ (0.025384657508457316) [X2 Y3 Y10 X11]
+ (0.0286851837161059) [Y10 X11 X12 Y13]
+ (0.0286851837161059) [X10 Y11 Y12 X13]
+ (0.029812424517345906) [Y6 Z7 Z8 Y10]
+ (0.029812424517345906) [X6 Z7 Z8 X10]
+ (0.029812424517345906) [Y7 Z9 Z10 Y11]
+ (0.029812424517345906) [X7 Z9 Z10 X11]
+ (0.03010462314345691) [Y6 Z7 Z9 Y10]
+ (0.03010462314345691) [X6 Z7 Z9 X10]
+ (0.03010462314345691) [Y7 Z8 Z10 Y11]
+ (0.03010462314345691) [X7 Z8 Z10 X11]
+ (0.030787505389143988) [Y6 Z8 Z9 Y10]
+ (0.030787505389143988) [X6 Z8 Z9 X10]
+ (0.031143817988967207) [Y2 X3 X6 Y7]
+ (0.031143817988967207) [X2 Y3 Y6 X7]
+ (0.03583956795335342) [Y2 X3 X4 Y5]
+ (0.03583956795335342) [X2 Y3 Y4 X5]
+ (0.03619412355904273) [Y2 X3 X8 Y9]
+ (0.03619412355904273) [X2 Y3 Y8 X9]
+ (0.038314670294803864) [Y4 X5 X12 Y13]
+ (0.038314670294803864) [X4 Y5 Y12 X13]
+ (0.10433064780651433) [Z0 Y1 Z2 Y3]
+ (0.10433064780651433) [Z0 X1 Z2 X3]
+ (-0.12133276911042269) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042269) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042266) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042266) [X3 Z4 Z5 Z6 X7]
+ (3.2020768801240126e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768801240126e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768801240135e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768801240135e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918946) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918946) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918969) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918969) [X6 Z7 Z8 Z9 X10]
+ (-0.03276765782329046) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329046) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329046) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329046) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273114) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273114) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273114) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273114) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021093) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021093) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964615) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964615) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964615) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964615) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172987) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172987) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172987) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172987) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997614007) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997614007) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997614007) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997614007) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997614007) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997614007) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997614007) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997614007) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819215) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819215) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819215) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819215) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688692) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688692) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688692) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688692) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688692) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688692) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688692) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688692) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381041) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381041) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832998) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832998) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832998) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832998) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826933) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826933) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826933) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826933) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017345) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017345) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017345) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017345) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.0051433917688251535) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0051433917688251535) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.0051433917688251535) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.0051433917688251535) [X2 Z3 X4 X5 Z6 X7]
+ (-0.00468490338815522) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.00468490338815522) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.0046686203187763) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.0046686203187763) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639216) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639216) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441854) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441854) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840091) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840091) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840091) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840091) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890159) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890159) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890159) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890159) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255232) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255232) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524676) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524676) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630216) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630216) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369644) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369644) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730491) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730491) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730491) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730491) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125376) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125376) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956685) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956685) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956685) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956685) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880588465e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880588465e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880588465e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880588465e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.77481786434032e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77481786434032e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77481786434032e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77481786434032e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622154806486e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622154806486e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622154806486e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622154806486e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446757355714e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446757355714e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446757355714e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446757355714e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848179843e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848179843e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848179843e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848179843e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433106139e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433106139e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433106139e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433106139e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.97431171314699e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.97431171314699e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831218647154e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831218647154e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068313862e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068313862e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217988822e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217988822e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225434328e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225434328e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594518474273e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594518474273e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132944260314e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132944260314e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971304366035e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971304366035e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971304366035e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971304366035e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145499907672e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145499907672e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483195411733e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483195411733e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483195411733e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483195411733e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283482721714e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283482721714e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283482721714e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283482721714e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311028945e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311028945e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711200857e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711200857e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101345319e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101345319e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489099769e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489099769e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886047486e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886047486e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823745083e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823745083e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600241823e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600241823e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371575605e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371575605e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742118848e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742118848e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742118848e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742118848e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420114127e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420114127e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914372734e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914372734e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914372734e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914372734e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574334858e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574334858e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574334858e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574334858e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082597036e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082597036e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082597036e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082597036e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851910962587e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851910962587e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624384085e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624384085e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624384085e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624384085e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624384085e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624384085e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624384085e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624384085e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750494092e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750494092e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328686578e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328686578e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350248701e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350248701e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826564794153e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826564794153e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826564794153e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826564794153e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231285954846e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231285954846e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289476526764e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289476526764e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289476526764e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289476526764e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516184031787e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516184031787e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412769306473e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412769306473e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412769306473e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412769306473e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209152682573e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209152682573e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209152682573e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209152682573e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175702397e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175702397e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175702397e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175702397e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781479541374e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781479541374e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781479541374e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781479541374e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781479541374e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781479541374e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781479541374e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781479541374e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781479541374e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781479541374e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781479541374e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781479541374e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861053755e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861053755e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599474534e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599474534e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599474534e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599474534e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599474534e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599474534e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599474534e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599474534e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.05744659521812e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.05744659521812e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.05744659521812e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.05744659521812e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134226154e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134226154e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134226154e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134226154e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209152682573e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209152682573e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209152682573e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209152682573e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516184031787e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516184031787e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231285954846e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231285954846e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599610752917e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599610752917e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599610752917e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599610752917e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350248701e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350248701e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328686578e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328686578e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750494092e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750494092e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851910962587e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851910962587e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420114127e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420114127e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371575605e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371575605e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651447632e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651447632e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651447632e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651447632e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600241823e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600241823e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823745083e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823745083e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216241784e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216241784e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216241784e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216241784e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886047486e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886047486e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489099769e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489099769e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101345319e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101345319e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711200857e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711200857e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479457665256e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479457665256e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311028945e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311028945e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145499907672e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145499907672e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312891350805e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312891350805e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132944260314e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132944260314e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559215436e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559215436e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217988822e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217988822e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068313862e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068313862e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831218647154e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831218647154e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.97431171314699e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.97431171314699e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611100736) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611100736) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611100736) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611100736) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915417) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915417) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499326) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499326) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499326) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499326) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125376) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125376) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213802) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213802) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213802) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213802) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667604181144056) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667604181144056) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667604181144056) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667604181144056) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369644) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369644) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630216) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630216) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524676) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524676) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133918) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133918) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133918) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133918) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496524) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496524) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496524) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496524) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441854) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441854) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639216) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639216) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.0046686203187763) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.0046686203187763) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.00468490338815522) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.00468490338815522) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221703) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221703) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221703) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221703) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109636) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109636) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109636) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109636) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921571) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921571) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921571) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921571) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381041) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381041) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269462) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269462) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269462) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269462) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.0102634148681585) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.0102634148681585) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.0102634148681585) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.0102634148681585) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671595) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671595) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671595) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671595) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542628) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542628) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542628) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542628) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848185) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848185) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130893) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130893) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130893) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130893) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226563) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226563) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226563) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226563) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380205) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380205) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380205) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380205) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375626) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375626) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375626) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375626) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317304005) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317304005) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317304005) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317304005) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535578) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535578) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535578) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535578) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535578) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535578) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535578) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535578) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068893) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068893) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068893) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068893) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068893) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068893) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068893) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068893) [X3 Z4 X5 X10 Z11 X12]
+ (0.025104957138844582) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844582) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844582) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844582) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143988) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143988) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297824) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297824) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807834) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807834) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807834) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807834) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661372) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661372) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661372) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661372) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928260501e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928260501e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928260498e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928260498e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860068894556e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860068894556e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086006889455e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006889455e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013783054) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783054) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783054) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783054) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638307) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638307) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638307) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638307) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821726) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821726) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821726) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821726) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289322) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289322) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289322) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289322) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719749) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719749) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719749) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719749) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831258) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831258) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905502) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905502) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905502) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905502) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026803) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026803) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026803) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026803) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890925) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890925) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890925) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890925) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692902) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692902) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529016) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529016) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601293) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601293) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160082) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160082) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160082) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160082) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251575) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251575) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847192) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847192) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942867) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942867) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942867) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942867) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0160246036891796) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.0160246036891796) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226563) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226563) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162071) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162071) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172987) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172987) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819215) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819215) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840957) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840957) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962633) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962633) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0096126346068472) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.0096126346068472) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.0096126346068472) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.0096126346068472) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102396) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102396) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832998) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832998) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561342) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561342) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017345) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017345) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109636) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109636) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840091) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840091) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328714) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328714) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328714) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328714) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235394) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235394) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235394) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235394) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255232) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255232) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066063) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066063) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066063) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066063) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524676) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524676) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524676) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524676) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696419) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696419) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696419) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696419) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696419) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696419) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696419) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696419) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569573163) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569573163) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551355) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303551355) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303551355) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303551355) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880588465e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880588465e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305197225e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305197225e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305197225e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305197225e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879482041e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879482041e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879482041e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879482041e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277484442e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277484442e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277484442e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277484442e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467317476e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467317476e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467317476e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467317476e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669281475e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669281475e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669281475e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669281475e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183376576e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.48185183376576e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183376576e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.48185183376576e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736219279e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736219279e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736219279e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736219279e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.73462203862514e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.73462203862514e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.73462203862514e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.73462203862514e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147111071e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147111071e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147111071e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147111071e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.2532242254343275e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.2532242254343275e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594518474273e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594518474273e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954291296246e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954291296246e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954291296246e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954291296246e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954291296246e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954291296246e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954291296246e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954291296246e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202064044e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202064044e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202064044e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202064044e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045885387e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045885387e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045885387e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045885387e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098146927e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098146927e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098146927e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098146927e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365542807e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365542807e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365542807e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365542807e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770452476e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174770452476e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770452476e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174770452476e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.522493067576598e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.522493067576598e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.522493067576598e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.522493067576598e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.522493067576598e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067576598e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067576598e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.522493067576598e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823745083e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823745083e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823745083e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823745083e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770287642009e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770287642009e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770287642009e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770287642009e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103768122e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103768122e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103768122e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103768122e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974984315e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974984315e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206683713e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206683713e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744428599e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744428599e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471792876746e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471792876746e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471792876746e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471792876746e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677783899e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677783899e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108354335e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108354335e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108354335e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108354335e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350248701e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350248701e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350248701e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350248701e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826564794153e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826564794153e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935950903295e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935950903295e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935950903295e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935950903295e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289476526764e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289476526764e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209152682575e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209152682575e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595218122e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595218122e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096261884e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096261884e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096261884e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096261884e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595218122e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595218122e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350644161158e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644161158e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644161158e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350644161158e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355157143e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355157143e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355157143e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355157143e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209152682575e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209152682575e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289476526764e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289476526764e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826564794153e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826564794153e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677783899e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677783899e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744428599e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744428599e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206683713e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206683713e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974984315e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974984315e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886047486e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886047486e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886047486e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886047486e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434523466e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434523466e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434523466e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434523466e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951402876e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951402876e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951402876e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951402876e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003654236e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003654236e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003654236e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003654236e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003654236e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003654236e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003654236e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003654236e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420189794742e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189794742e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189794742e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189794742e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189794742e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189794742e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420189794742e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189794742e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145499907672e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145499907672e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145499907672e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145499907672e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312891350805e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312891350805e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559215436e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559215436e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880588465e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880588465e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569573163) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569573163) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409236) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409236) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409236) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409236) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005567) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005567) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005567) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005567) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005567) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005567) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005567) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005567) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125377) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125377) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125377) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125377) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907592) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907592) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907592) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907592) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496712) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496712) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496712) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496712) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126977) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126977) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126977) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126977) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823394) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823394) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823394) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823394) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823394) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823394) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823394) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823394) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619303) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619303) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619303) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619303) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840091) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840091) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914298) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914298) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914298) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914298) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182543) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182543) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182543) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182543) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660392) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660392) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660392) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660392) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660392) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660392) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660392) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660392) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803848) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803848) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803848) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803848) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076857) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076857) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076857) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076857) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109636) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109636) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839363) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839363) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839363) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839363) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017345) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017345) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960949) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960949) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960949) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960949) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561342) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561342) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832998) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832998) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102396) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102396) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962633) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962633) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840957) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840957) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819215) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819215) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172987) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172987) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162071) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162071) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226563) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226563) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.0160246036891796) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.0160246036891796) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847192) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847192) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251575) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251575) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297824) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297824) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615617) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615617) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615617) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615617) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702269) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702269) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022677) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022677) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036467) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036467) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036467) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036467) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863614) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863614) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863614) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863614) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634993) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634993) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634993) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634993) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214007) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214007) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214007) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214007) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831259) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831259) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366193) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366193) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366193) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366193) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830006) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830006) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830006) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830006) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0242821173546929) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0242821173546929) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952902) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952902) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601293) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601293) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314645) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314645) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314645) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314645) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898845) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898845) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898845) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898845) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0160246036891796) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.0160246036891796) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.0160246036891796) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.0160246036891796) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831924) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831924) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831924) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831924) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962631) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962631) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962631) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962631) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209842) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209842) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209842) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209842) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454792) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454792) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454792) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454792) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454792) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454792) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454792) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454792) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102396) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102396) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102396) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102396) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.0046686203187763) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0046686203187763) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00387647089933693) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.00387647089933693) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728536) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728536) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021789) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021789) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832872) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832872) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235394) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235394) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015655) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015655) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369644) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369644) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124267) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124267) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416882) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416882) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416882) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416882) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024355) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024355) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487707) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487707) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756548) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756548) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551355) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303551355) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153385e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153385e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153385e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153385e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736219279e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736219279e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311028945e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311028945e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711200857e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711200857e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117060550023e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117060550023e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071218303e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071218303e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202064044e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202064044e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561310493e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561310493e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650711318e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650711318e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650711318e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650711318e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102823905e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102823905e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102823905e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102823905e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198823758e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198823758e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198823758e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198823758e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198823758e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198823758e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198823758e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198823758e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985655909e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985655909e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985655909e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985655909e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.90012898607332e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.90012898607332e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.90012898607332e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.90012898607332e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103768122e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103768122e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464542465e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464542465e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464542465e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464542465e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464542465e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464542465e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464542465e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464542465e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422009269e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422009269e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422009269e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422009269e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422009269e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422009269e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422009269e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422009269e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521039859e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521039859e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521039859e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521039859e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082894206e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393082894206e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082894206e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393082894206e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082894206e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393082894206e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393082894206e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393082894206e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.88829359509033e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.88829359509033e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815463424417e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815463424417e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355157143e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355157143e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350644161158e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350644161158e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244765179e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244765179e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244765179e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244765179e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244765179e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244765179e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244765179e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244765179e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253795043584e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253795043584e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253795043584e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253795043584e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716556132129e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716556132129e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716556132129e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716556132129e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350644161158e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350644161158e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184025981e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184025981e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184025981e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184025981e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287492914493e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287492914493e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287492914493e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287492914493e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355157143e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355157143e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943050467148e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943050467148e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943050467148e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943050467148e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815463424417e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815463424417e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.88829359509033e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.88829359509033e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506159155574e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506159155574e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506159155574e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506159155574e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506159155574e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506159155574e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506159155574e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506159155574e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978538176374e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978538176374e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978538176374e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978538176374e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915094946208e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915094946208e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915094946208e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915094946208e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425086311e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425086311e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425086311e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425086311e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425086311e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425086311e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425086311e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425086311e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103768122e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103768122e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561310493e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561310493e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202064044e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202064044e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071218303e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071218303e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.88367657581573e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.88367657581573e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011529342e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011529342e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011529342e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011529342e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117060550023e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117060550023e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711200857e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711200857e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311028945e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311028945e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671026427e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671026427e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671026427e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671026427e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736219279e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736219279e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721658793e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721658793e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721658793e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721658793e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327157476e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327157476e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327157476e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327157476e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501540797e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501540797e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501540797e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501540797e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.42798865617504e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.42798865617504e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.42798865617504e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.42798865617504e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717584345e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717584345e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717584345e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717584345e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347645182e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347645182e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825792877096e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825792877096e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825792877096e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825792877096e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216099e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411216099e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411216099e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216099e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303551355) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303551355) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389554442) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389554442) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389554442) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389554442) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756548) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756548) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569573163) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569573163) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569573163) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569573163) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487707) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487707) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909017) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909017) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909017) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909017) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024355) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024355) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730643) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730643) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730643) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730643) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124267) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124267) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369644) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369644) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158054) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158054) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158054) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158054) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235394) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235394) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832872) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832872) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.00348415730021789) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00348415730021789) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00387647089933693) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.00387647089933693) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.0046686203187763) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0046686203187763) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278109) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278109) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278109) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278109) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865465382268785) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865465382268785) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865465382268785) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865465382268785) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409994) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409994) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409994) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409994) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796782) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796782) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796782) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796782) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908944) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908944) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908944) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908944) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162071) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162071) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162071) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162071) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936374) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936374) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936374) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936374) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936374) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936374) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936374) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936374) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386173) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386173) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950526807002e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950526807002e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950526807003e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950526807003e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002687) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002687) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002687) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002687) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251575) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251575) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831924) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831924) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209842) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209842) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297706226) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297706226) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297706226) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297706226) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311873) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311873) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311873) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311873) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311873) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311873) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311873) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311873) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676637) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676637) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676637) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676637) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728536) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728536) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219178) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219178) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219178) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219178) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415805) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415805) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939835) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939835) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939835) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939835) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015655) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015655) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587492) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587492) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587492) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587492) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587492) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587492) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587492) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587492) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124267) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124267) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124267) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124267) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538227) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538227) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538227) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538227) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538227) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538227) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538227) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538227) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562572) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562572) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562572) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562572) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452225692e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452225692e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071218303e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071218303e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071218303e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071218303e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561310493e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561310493e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561310493e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561310493e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297032276e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297032276e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297032276e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297032276e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95607922908063e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95607922908063e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95607922908063e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95607922908063e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036409264e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036409264e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036409264e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036409264e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212470192e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212470192e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212470192e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212470192e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413283383e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413283383e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974984315e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974984315e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657635999e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657635999e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657635999e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657635999e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206683713e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206683713e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677783899e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677783899e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325314086253e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325314086253e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325314086253e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325314086253e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714587170247e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714587170247e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998837488935e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998837488935e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998837488935e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998837488935e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317541474515e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317541474515e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317541474515e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317541474515e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641926713637e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641926713637e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309319145787e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309319145787e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309319145787e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309319145787e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641926713637e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641926713637e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815463424417e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815463424417e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815463424417e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815463424417e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587170247e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714587170247e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677783899e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677783899e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390631604e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390631604e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390631604e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390631604e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206683713e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206683713e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974984315e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974984315e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413283383e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413283383e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487178183e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487178183e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939575897467e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939575897467e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939575897467e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939575897467e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.88367657581573e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.88367657581573e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117060550023e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117060550023e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117060550023e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117060550023e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347645182e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347645182e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734431687e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734431687e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734431687e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734431687e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692021432e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692021432e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692021432e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692021432e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487707) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487707) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487707) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487707) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024355) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024355) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024355) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024355) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441894) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441894) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441894) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441894) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245584) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245584) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245584) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245584) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500446) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500446) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500446) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500446) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798012) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798012) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798012) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798012) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798012) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798012) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798012) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798012) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415805) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415805) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728536) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728536) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00387647089933693) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.00387647089933693) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.00387647089933693) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.00387647089933693) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046477) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046477) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046477) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046477) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209842) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209842) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831924) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831924) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251575) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251575) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386173) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386173) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009012405578e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009012405578e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009012405575e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009012405575e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021789) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021789) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219178) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219178) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756548) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756548) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452225692e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452225692e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939575897467e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939575897467e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413283383e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413283383e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413283383e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413283383e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641926713637e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641926713637e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641926713637e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641926713637e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458717025e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458717025e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458717025e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458717025e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487178182e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487178182e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939575897467e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939575897467e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756548) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756548) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219178) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219178) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.00348415730021789) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00348415730021789) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352531) [I0]
+ (-0.18066792656583372) [Z7]
+ (-0.18066792656583364) [Z6]
+ (-0.1596143250180982) [Z5]
+ (-0.15961432501809814) [Z4]
+ (0.17419956155055744) [Z2]
+ (0.17419956155055744) [Z3]
+ (0.22757269005453537) [Z0]
+ (0.2275726900545355) [Z1]
+ (-8.194261372278763e-06) [Y4 Y6]
+ (-8.194261372278763e-06) [X4 X6]
+ (7.954413176367876e-06) [Y5 Y7]
+ (7.954413176367876e-06) [X5 X7]
+ (0.11270386920332214) [Z4 Z6]
+ (0.11270386920332214) [Z5 Z7]
+ (0.11952438964682671) [Z0 Z4]
+ (0.11952438964682671) [Z1 Z5]
+ (0.13401715261963704) [Z0 Z6]
+ (0.13401715261963704) [Z1 Z7]
+ (0.13734953064261318) [Z0 Z5]
+ (0.13734953064261318) [Z1 Z4]
+ (0.13766872645852574) [Z2 Z4]
+ (0.13766872645852574) [Z3 Z5]
+ (0.141389052919428) [Z4 Z7]
+ (0.141389052919428) [Z5 Z6]
+ (0.14722943218766166) [Z2 Z5]
+ (0.14722943218766166) [Z3 Z4]
+ (0.14926355147388887) [Z4 Z5]
+ (0.14973486803496927) [Z2 Z6]
+ (0.14973486803496927) [Z3 Z7]
+ (0.15138327161428838) [Z0 Z7]
+ (0.15138327161428838) [Z1 Z6]
+ (0.15435748657223625) [Z6 Z7]
+ (0.1558226905155311) [Z2 Z7]
+ (0.1558226905155311) [Z3 Z6]
+ (0.16756653265461266) [Z0 Z2]
+ (0.16756653265461266) [Z1 Z3]
+ (0.18143991440303872) [Z0 Z3]
+ (0.18143991440303872) [Z1 Z2]
+ (0.19392534613270188) [Z0 Z1]
+ (0.2200397733437609) [Z2 Z3]
+ (-7.0378875106018e-06) [Y4 Z5 Y6]
+ (-7.0378875106018e-06) [X4 Z5 X6]
+ (-7.037887510601798e-06) [Y5 Z6 Y7]
+ (-7.037887510601798e-06) [X5 Z6 X7]
+ (-0.028685183716105872) [Y4 Y5 X6 X7]
+ (-0.028685183716105872) [X4 X5 Y6 Y7]
+ (-0.017825140995786467) [Y0 Y1 X4 X5]
+ (-0.017825140995786467) [X0 X1 Y4 Y5]
+ (-0.017366118994651375) [Y0 Y1 X6 X7]
+ (-0.017366118994651375) [X0 X1 Y6 Y7]
+ (-0.013873381748426087) [Y0 Y1 X2 X3]
+ (-0.013873381748426087) [X0 X1 Y2 Y3]
+ (-0.009560705729135949) [Y2 Y3 X4 X5]
+ (-0.009560705729135949) [X2 X3 Y4 Y5]
+ (-0.00608782248056186) [Y2 Y3 X6 X7]
+ (-0.00608782248056186) [X2 X3 Y6 Y7]
+ (-0.0002921986261110532) [Y1 Y2 X3 X4]
+ (-0.0002921986261110532) [X1 X2 Y3 Y4]
+ (-8.194261372278763e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372278763e-06) [Z4 X5 Z6 X7]
+ (-2.8909678816509188e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678816509188e-06) [Z0 X5 Z6 X7]
+ (-2.8909678816509188e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678816509188e-06) [Z1 X4 Z5 X6]
+ (-1.8551201214066286e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201214066286e-06) [Z0 X4 Z5 X6]
+ (-1.8551201214066286e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201214066286e-06) [Z1 X5 Z6 X7]
+ (-1.5973171976888542e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171976888542e-06) [Z2 X4 Z5 X6]
+ (-1.5973171976888542e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171976888542e-06) [Z3 X5 Z6 X7]
+ (-1.03584776024429e-06) [Y0 X1 X5 Y6]
+ (-1.03584776024429e-06) [Y0 Y1 Y5 Y6]
+ (-1.03584776024429e-06) [X0 X1 X5 X6]
+ (-1.03584776024429e-06) [X0 Y1 Y5 X6]
+ (-9.344557774971821e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557774971821e-07) [Z2 X5 Z6 X7]
+ (-9.344557774971821e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557774971821e-07) [Z3 X4 Z5 X6]
+ (6.62861420191672e-07) [Y2 X3 X5 Y6]
+ (6.62861420191672e-07) [Y2 Y3 Y5 Y6]
+ (6.62861420191672e-07) [X2 X3 X5 X6]
+ (6.62861420191672e-07) [X2 Y3 Y5 X6]
+ (7.954413176367876e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176367876e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110532) [Y1 X2 X3 Y4]
+ (0.0002921986261110532) [X1 Y2 Y3 X4]
+ (0.00608782248056186) [Y2 X3 X6 Y7]
+ (0.00608782248056186) [X2 Y3 Y6 X7]
+ (0.009560705729135949) [Y2 X3 X4 Y5]
+ (0.009560705729135949) [X2 Y3 Y4 X5]
+ (0.011307274008848227) [Y1 Z2 Z3 Y5]
+ (0.011307274008848227) [X1 Z2 Z3 X5]
+ (0.013873381748426087) [Y0 X1 X2 Y3]
+ (0.013873381748426087) [X0 Y1 Y2 X3]
+ (0.017366118994651375) [Y0 X1 X6 Y7]
+ (0.017366118994651375) [X0 Y1 Y6 X7]
+ (0.017825140995786467) [Y0 X1 X4 Y5]
+ (0.017825140995786467) [X0 Y1 Y4 X5]
+ (0.028685183716105872) [Y4 X5 X6 Y7]
+ (0.028685183716105872) [X4 Y5 Y6 X7]
+ (0.02981242451734579) [Y0 Z1 Z2 Y4]
+ (0.02981242451734579) [X0 Z1 Z2 X4]
+ (0.02981242451734579) [Y1 Z3 Z4 Y5]
+ (0.02981242451734579) [X1 Z3 Z4 X5]
+ (0.030787505389143925) [Y0 Z2 Z3 Y4]
+ (0.030787505389143925) [X0 Z2 Z3 X4]
+ (0.043752638010660316) [Y1 Z2 Z3 Z4 Y5]
+ (0.043752638010660316) [X1 Z2 Z3 Z4 X5]
+ (0.04375263801066032) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066032) [X0 Z1 Z2 Z3 X4]
+ (-0.01456453123117298) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.01456453123117298) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.01456453123117298) [X1 Z2 Z3 X4 X6 X7]
+ (-0.01456453123117298) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848620186e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848620186e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848620186e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848620186e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594518394656e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594518394656e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971304433463e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971304433463e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971304433463e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971304433463e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455002964293e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455002964293e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831953561567e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831953561567e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831953561567e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831953561567e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283483237574e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283483237574e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283483237574e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283483237574e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477602442902e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477602442902e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201916721e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201916721e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393508718977e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393508718977e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393508718977e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393508718977e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201916721e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201916721e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477602442902e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477602442902e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455002964293e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455002964293e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559517688e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559517688e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0002921986261110532) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002921986261110532) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002921986261110532) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002921986261110532) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671564) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671564) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671564) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671564) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848227) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848227) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844544) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844544) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844544) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844544) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143925) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143925) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549474157e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549474157e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396549474157e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549474157e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.01456453123117298) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.01456453123117298) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594518394656e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594518394656e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393508718977e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393508718977e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393508718977e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393508718977e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455002964293e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455002964293e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455002964293e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455002964293e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559517689e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559517689e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.01456453123117298) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.01456453123117298) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 31. tutorial_jax_transformations.html <a name="demo30"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
Result: DeviceArray(0.99244501, dtype=float64)
No jit time: 0.0105 seconds
First run time: 0.0576 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
Result: DeviceArray(0.99244503, dtype=float64)
No jit time: 0.0090 seconds
First run time: 0.0612 seconds
```

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
   (-46.46390678868896) [I0]
+ (0.7829661725950214) [Z11]
+ (0.7829661725950215) [Z10]
+ (0.8084581961720482) [Z12]
+ (0.8084581961720491) [Z13]
+ (1.2034402289145627) [Z4]
+ (1.2034402289145627) [Z5]
+ (1.309686298861544) [Z7]
+ (1.3096862988615445) [Z6]
+ (1.3693525634718189) [Z8]
+ (1.3693525634718193) [Z9]
+ (1.653894222683168) [Z2]
+ (1.6538942226831685) [Z3]
+ (-8.194261372193745e-06) [Y10 Y12]
+ (-8.194261372193745e-06) [X10 X12]
+ (-1.8540608580880267e-06) [Y5 Y7]
+ (-1.8540608580880267e-06) [X5 X7]
+ (-7.764994120189402e-07) [Y3 Y5]
+ (-7.764994120189402e-07) [X3 X5]
+ (-5.929765815918017e-07) [Y4 Y6]
+ (-5.929765815918017e-07) [X4 X6]
+ (1.602116740739445e-06) [Y2 Y4]
+ (1.602116740739445e-06) [X2 X4]
+ (7.954413176224722e-06) [Y11 Y13]
+ (7.954413176224722e-06) [X11 X13]
+ (0.0032769719312316864) [Y1 Y3]
+ (0.0032769719312316864) [X1 X3]
+ (0.1127038692033221) [Z10 Z12]
+ (0.1127038692033221) [Z11 Z13]
+ (0.11383573679388634) [Z4 Z12]
+ (0.11383573679388634) [Z5 Z13]
+ (0.11952438964682668) [Z6 Z10]
+ (0.11952438964682668) [Z7 Z11]
+ (0.12489990917237577) [Z4 Z10]
+ (0.12489990917237577) [Z5 Z11]
+ (0.12495807739503169) [Z2 Z4]
+ (0.12495807739503169) [Z3 Z5]
+ (0.1279950249246839) [Z2 Z10]
+ (0.1279950249246839) [Z3 Z11]
+ (0.1340171526196369) [Z6 Z12]
+ (0.1340171526196369) [Z7 Z13]
+ (0.13701191674040716) [Z4 Z6]
+ (0.13701191674040716) [Z5 Z7]
+ (0.1373495306426131) [Z6 Z11]
+ (0.1373495306426131) [Z7 Z10]
+ (0.13739104762683194) [Z2 Z6]
+ (0.13739104762683194) [Z3 Z7]
+ (0.13766872645852576) [Z8 Z10]
+ (0.13766872645852576) [Z9 Z11]
+ (0.1401128986535478) [Z2 Z12]
+ (0.1401128986535478) [Z3 Z13]
+ (0.14138905291942788) [Z10 Z13]
+ (0.14138905291942788) [Z11 Z12]
+ (0.14257997712485726) [Z4 Z11]
+ (0.14257997712485726) [Z5 Z10]
+ (0.1472294321876617) [Z8 Z11]
+ (0.1472294321876617) [Z9 Z10]
+ (0.1489943057506551) [Z4 Z7]
+ (0.1489943057506551) [Z5 Z6]
+ (0.14926355147388895) [Z10 Z11]
+ (0.14960702684445268) [Z4 Z8]
+ (0.14960702684445268) [Z5 Z9]
+ (0.14973486803496924) [Z8 Z12]
+ (0.14973486803496924) [Z9 Z13]
+ (0.15071408121008262) [Z2 Z8]
+ (0.15071408121008262) [Z3 Z9]
+ (0.15138327161428833) [Z6 Z13]
+ (0.15138327161428833) [Z7 Z12]
+ (0.1521504070886901) [Z4 Z13]
+ (0.1521504070886901) [Z5 Z12]
+ (0.15337968243314132) [Z2 Z11]
+ (0.15337968243314132) [Z3 Z10]
+ (0.15435748657223614) [Z12 Z13]
+ (0.15569010671752426) [Z2 Z13]
+ (0.15569010671752426) [Z3 Z12]
+ (0.1558226905155311) [Z8 Z13]
+ (0.1558226905155311) [Z9 Z12]
+ (0.1567639617643096) [Z4 Z9]
+ (0.1567639617643096) [Z5 Z8]
+ (0.15755314797985615) [Z4 Z5]
+ (0.1607976453483851) [Z2 Z5]
+ (0.1607976453483851) [Z3 Z4]
+ (0.16756653265461258) [Z6 Z8]
+ (0.16756653265461258) [Z7 Z9]
+ (0.16853486561579895) [Z2 Z7]
+ (0.16853486561579895) [Z3 Z6]
+ (0.1814399144030387) [Z6 Z9]
+ (0.1814399144030387) [Z7 Z8]
+ (0.18189085790751303) [Z2 Z3]
+ (0.18690820476912512) [Z2 Z9]
+ (0.18690820476912512) [Z3 Z8]
+ (0.1929972393536425) [Z0 Z10]
+ (0.1929972393536425) [Z1 Z11]
+ (0.1939253461327019) [Z6 Z7]
+ (0.19661770890342095) [Z0 Z4]
+ (0.19661770890342095) [Z1 Z5]
+ (0.19936354537360776) [Z0 Z5]
+ (0.19936354537360776) [Z1 Z4]
+ (0.20072866460441777) [Z0 Z11]
+ (0.20072866460441777) [Z1 Z10]
+ (0.21102659849791516) [Z0 Z12]
+ (0.21102659849791516) [Z1 Z13]
+ (0.2163103749863181) [Z0 Z13]
+ (0.2163103749863181) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830367) [Z0 Z2]
+ (0.23671080783830367) [Z1 Z3]
+ (0.24164663936017192) [Z0 Z6]
+ (0.24164663936017192) [Z1 Z7]
+ (0.25129445674591633) [Z0 Z3]
+ (0.25129445674591633) [Z1 Z2]
+ (0.2723251830660568) [Z0 Z8]
+ (0.2723251830660568) [Z1 Z9]
+ (0.2788345442672341) [Z0 Z9]
+ (0.2788345442672341) [Z1 Z8]
+ (1.1861763734860495) [Z0 Z1]
+ (-1.2260484989516684e-05) [Y4 Z5 Y6]
+ (-1.2260484989516684e-05) [X4 Z5 X6]
+ (-1.2260484989516683e-05) [Y5 Z6 Y7]
+ (-1.2260484989516683e-05) [X5 Z6 X7]
+ (-1.0722312157827875e-05) [Y11 Z12 Y13]
+ (-1.0722312157827875e-05) [X11 Z12 X13]
+ (-1.0722312157827863e-05) [Y10 Z11 Y12]
+ (-1.0722312157827863e-05) [X10 Z11 X12]
+ (-3.88705167503564e-06) [Y3 Z4 Y5]
+ (-3.88705167503564e-06) [X3 Z4 X5]
+ (-3.887051675035638e-06) [Y2 Z3 Y4]
+ (-3.887051675035638e-06) [X2 Z3 X4]
+ (0.12507032579772215) [Y1 Z2 Y3]
+ (0.12507032579772215) [X1 Z2 X3]
+ (0.12507032579772218) [Y0 Z1 Y2]
+ (0.12507032579772218) [X0 Z1 X2]
+ (-0.038314670294803795) [Y4 Y5 X12 X13]
+ (-0.038314670294803795) [X4 X5 Y12 Y13]
+ (-0.03619412355904251) [Y2 Y3 X8 X9]
+ (-0.03619412355904251) [X2 X3 Y8 Y9]
+ (-0.03583956795335338) [Y2 Y3 X4 X5]
+ (-0.03583956795335338) [X2 X3 Y4 Y5]
+ (-0.031143817988967013) [Y2 Y3 X6 X7]
+ (-0.031143817988967013) [X2 X3 Y6 Y7]
+ (-0.028685183716105792) [Y10 Y11 X12 X13]
+ (-0.028685183716105792) [X10 X11 Y12 Y13]
+ (-0.025996177598021263) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021263) [X3 Z4 Z5 X7]
+ (-0.02538465750845744) [Y2 Y3 X10 X11]
+ (-0.02538465750845744) [X2 X3 Y10 Y11]
+ (-0.01902824244384729) [Y3 Y4 X11 X12]
+ (-0.01902824244384729) [X3 X4 Y11 Y12]
+ (-0.01782514099578642) [Y6 Y7 X10 X11]
+ (-0.01782514099578642) [X6 X7 Y10 Y11]
+ (-0.017680067952481487) [Y4 Y5 X10 X11]
+ (-0.017680067952481487) [X4 X5 Y10 Y11]
+ (-0.015577208063976444) [Y2 Y3 X12 X13]
+ (-0.015577208063976444) [X2 X3 Y12 Y13]
+ (-0.01458364890761266) [Y0 Y1 X2 X3]
+ (-0.01458364890761266) [X0 X1 Y2 Y3]
+ (-0.013873381748426129) [Y6 Y7 X8 X9]
+ (-0.013873381748426129) [X6 X7 Y8 Y9]
+ (-0.01198238901024793) [Y4 Y5 X6 X7]
+ (-0.01198238901024793) [X4 X5 Y6 Y7]
+ (-0.011285190200840867) [Y5 X6 X11 Y12]
+ (-0.011285190200840867) [X5 Y6 Y11 X12]
+ (-0.009560705729135975) [Y8 Y9 X10 X11]
+ (-0.009560705729135975) [X8 X9 Y10 Y11]
+ (-0.008125251921381013) [Y1 X2 X8 Y9]
+ (-0.008125251921381013) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381013) [X1 X2 X8 X9]
+ (-0.008125251921381013) [X1 Y2 Y8 X9]
+ (-0.007731425250775283) [Y0 Y1 X10 X11]
+ (-0.007731425250775283) [X0 X1 Y10 Y11]
+ (-0.007156934919856923) [Y4 Y5 X8 X9]
+ (-0.007156934919856923) [X4 X5 Y8 Y9]
+ (-0.0068881943529705775) [Y0 Y1 X6 X7]
+ (-0.0068881943529705775) [X0 X1 Y6 Y7]
+ (-0.0065093612011772346) [Y0 Y1 X8 X9]
+ (-0.0065093612011772346) [X0 X1 Y8 Y9]
+ (-0.006087822480561868) [Y8 Y9 X12 X13]
+ (-0.006087822480561868) [X8 X9 Y12 Y13]
+ (-0.005283776488402962) [Y0 Y1 X12 X13]
+ (-0.005283776488402962) [X0 X1 Y12 Y13]
+ (-0.005143391768825117) [Y3 X4 X5 Y6]
+ (-0.005143391768825117) [X3 Y4 Y5 X6]
+ (-0.004684903388155181) [Y1 X2 X6 Y7]
+ (-0.004684903388155181) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155181) [X1 X2 X6 X7]
+ (-0.004684903388155181) [X1 Y2 Y6 X7]
+ (-0.0045750076266392) [Y1 X2 X12 Y13]
+ (-0.0045750076266392) [Y1 Y2 Y12 Y13]
+ (-0.0045750076266392) [X1 X2 X12 X13]
+ (-0.0045750076266392) [X1 Y2 Y12 X13]
+ (-0.004424855449441833) [Y1 X2 X4 Y5]
+ (-0.004424855449441833) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441833) [X1 X2 X4 X5]
+ (-0.004424855449441833) [X1 Y2 Y4 X5]
+ (-0.003479511890334343) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334343) [X2 Z3 Z5 X6]
+ (-0.003479511890334343) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334343) [X3 Z4 Z6 X7]
+ (-0.0027458364701867977) [Y0 Y1 X4 X5]
+ (-0.0027458364701867977) [X0 X1 Y4 Y5]
+ (-0.001799219493663029) [Y1 X2 X10 Y11]
+ (-0.001799219493663029) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663029) [X1 X2 X10 X11]
+ (-0.001799219493663029) [X1 Y2 Y10 X11]
+ (-0.0002921986261110854) [Y7 Y8 X9 X10]
+ (-0.0002921986261110854) [X7 X8 Y9 Y10]
+ (-8.194261372193745e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372193745e-06) [Z10 X11 Z12 X13]
+ (-7.801707500467112e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500467112e-06) [X2 Z3 X4 Z11]
+ (-7.801707500467112e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500467112e-06) [X3 Z4 X5 Z10]
+ (-4.643051068428096e-06) [Y3 X4 X10 Y11]
+ (-4.643051068428096e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068428096e-06) [X3 X4 X10 X11]
+ (-4.643051068428096e-06) [X3 Y4 Y10 X11]
+ (-4.588855155615013e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155615013e-06) [X4 Z5 X6 Z13]
+ (-4.588855155615013e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155615013e-06) [X5 Z6 X7 Z12]
+ (-4.556569217994415e-06) [Y5 X6 X12 Y13]
+ (-4.556569217994415e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217994415e-06) [X5 X6 X12 X13]
+ (-4.556569217994415e-06) [X5 Y6 Y12 X13]
+ (-3.694513294408585e-06) [Y4 X5 X11 Y12]
+ (-3.694513294408585e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294408585e-06) [X4 X5 X11 X12]
+ (-3.694513294408585e-06) [X4 Y5 Y11 X12]
+ (-3.3440815567762594e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815567762594e-06) [Z0 X5 Z6 X7]
+ (-3.3440815567762594e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815567762594e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320390163e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320390163e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320390163e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320390163e-06) [X3 Z4 X5 Z11]
+ (-3.099349243861863e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243861863e-06) [Z0 X4 Z5 X6]
+ (-3.099349243861863e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243861863e-06) [Z1 X5 Z6 X7]
+ (-2.8909678818204694e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678818204694e-06) [Z6 X11 Z12 X13]
+ (-2.8909678818204694e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678818204694e-06) [Z7 X10 Z11 X12]
+ (-2.177664604989974e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604989974e-06) [Z0 X10 Z11 X12]
+ (-2.177664604989974e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604989974e-06) [Z1 X11 Z12 X13]
+ (-1.8818501833443887e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501833443887e-06) [X4 Z5 X6 Z9]
+ (-1.8818501833443887e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501833443887e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214935308e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214935308e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214935308e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214935308e-06) [Z7 X11 Z12 X13]
+ (-1.8540608580880265e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608580880265e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697610951e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697610951e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697610951e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697610951e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285210134e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285210134e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285210134e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285210134e-06) [X5 Z6 X7 Z11]
+ (-1.6148794138384906e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794138384906e-06) [Z0 X11 Z12 X13]
+ (-1.6148794138384906e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794138384906e-06) [Z1 X10 Z11 X12]
+ (-1.5973171978085878e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171978085878e-06) [Z8 X10 Z11 X12]
+ (-1.5973171978085878e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171978085878e-06) [Z9 X11 Z12 X13]
+ (-1.4548424492664058e-06) [Y3 X4 X6 Y7]
+ (-1.4548424492664058e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424492664058e-06) [X3 X4 X6 X7]
+ (-1.4548424492664058e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081931013e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081931013e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081931013e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081931013e-06) [X5 Z6 X7 Z9]
+ (-1.1954890102117135e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890102117135e-06) [X2 Z3 X4 Z7]
+ (-1.1954890102117135e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890102117135e-06) [X3 Z4 X5 Z6]
+ (-1.1908508086857574e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508086857574e-06) [Z0 X3 Z4 X5]
+ (-1.1908508086857574e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508086857574e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370550565e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370550565e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370550565e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370550565e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423277903e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423277903e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423277903e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423277903e-06) [Z3 X11 Z12 X13]
+ (-1.0358477603269385e-06) [Y6 X7 X11 Y12]
+ (-1.0358477603269385e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477603269385e-06) [X6 X7 X11 X12]
+ (-1.0358477603269385e-06) [X6 Y7 Y11 X12]
+ (-9.509249751951306e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751951306e-07) [Z2 X4 Z5 X6]
+ (-9.509249751951306e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751951306e-07) [Z3 X5 Z6 X7]
+ (-9.344557776316114e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776316114e-07) [Z8 X11 Z12 X13]
+ (-9.344557776316114e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776316114e-07) [Z9 X10 Z11 X12]
+ (-8.337746757182814e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746757182814e-07) [Z0 X2 Z3 X4]
+ (-8.337746757182814e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746757182814e-07) [Z1 X3 Z4 X5]
+ (-7.956895373858177e-07) [Y3 X4 X8 Y9]
+ (-7.956895373858177e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373858177e-07) [X3 X4 X8 X9]
+ (-7.956895373858177e-07) [X3 Y4 Y8 X9]
+ (-7.764994120189402e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120189402e-07) [X2 Z3 X4 Z5]
+ (-5.929765815918017e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815918017e-07) [Z4 X5 Z6 X7]
+ (-5.770052996906785e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996906785e-07) [X2 Z3 X4 Z9]
+ (-5.770052996906785e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996906785e-07) [X3 Z4 X5 Z8]
+ (-5.471647744715338e-07) [Y1 Y2 X11 X12]
+ (-5.471647744715338e-07) [X1 X2 Y11 Y12]
+ (-4.838052751512874e-07) [Y5 X6 X8 Y9]
+ (-4.838052751512874e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751512874e-07) [X5 X6 X8 X9]
+ (-4.838052751512874e-07) [X5 Y6 Y8 X9]
+ (-3.570761329674761e-07) [Y0 X1 X3 Y4]
+ (-3.570761329674761e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329674761e-07) [X0 X1 X3 X4]
+ (-3.570761329674761e-07) [X0 Y1 Y3 X4]
+ (-2.4473231291439617e-07) [Y0 X1 X5 Y6]
+ (-2.4473231291439617e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231291439617e-07) [X0 X1 X5 X6]
+ (-2.4473231291439617e-07) [X0 Y1 Y5 X6]
+ (-2.1990516185992589e-07) [Y2 X3 X5 Y6]
+ (-2.1990516185992589e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516185992589e-07) [X2 X3 X5 X6]
+ (-2.1990516185992589e-07) [X2 Y3 Y5 X6]
+ (-1.9332412773458046e-07) [Y1 X2 X3 Y4]
+ (-1.9332412773458046e-07) [X1 Y2 Y3 X4]
+ (-1.2919694865388208e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694865388208e-07) [X1 Z2 Z3 X5]
+ (1.7379332625347344e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332625347344e-07) [X0 Z1 Z3 X4]
+ (1.7379332625347344e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332625347344e-07) [X1 Z2 Z4 X5]
+ (1.9332412773458046e-07) [Y1 Y2 X3 X4]
+ (1.9332412773458046e-07) [X1 X2 Y3 Y4]
+ (2.186842376951393e-07) [Y2 Z3 Y4 Z8]
+ (2.186842376951393e-07) [X2 Z3 X4 Z8]
+ (2.186842376951393e-07) [Y3 Z4 Y5 Z9]
+ (2.186842376951393e-07) [X3 Z4 X5 Z9]
+ (2.5935343905469227e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343905469227e-07) [X2 Z3 X4 Z6]
+ (2.5935343905469227e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343905469227e-07) [X3 Z4 X5 Z7]
+ (3.606071868274887e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868274887e-07) [X0 Z1 Z2 X4]
+ (3.606071868274887e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868274887e-07) [X1 Z3 Z4 X5]
+ (5.471647744715338e-07) [Y1 X2 X11 Y12]
+ (5.471647744715338e-07) [X1 Y2 Y11 X12]
+ (5.627851911514835e-07) [Y0 X1 X11 Y12]
+ (5.627851911514835e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911514835e-07) [X0 X1 X11 X12]
+ (5.627851911514835e-07) [X0 Y1 Y11 X12]
+ (6.628614201769763e-07) [Y8 X9 X11 Y12]
+ (6.628614201769763e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201769763e-07) [X8 X9 X11 X12]
+ (6.628614201769763e-07) [X8 Y9 Y11 X12]
+ (1.109440759263316e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759263316e-06) [Z2 X11 Z12 X13]
+ (1.109440759263316e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759263316e-06) [Z3 X10 Z11 X12]
+ (1.602116740739445e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740739445e-06) [Z2 X3 Z4 X5]
+ (1.8782101246474902e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101246474902e-06) [Z4 X10 Z11 X12]
+ (1.8782101246474902e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101246474902e-06) [Z5 X11 Z12 X13]
+ (2.172669101591106e-06) [Y2 X3 X11 Y12]
+ (2.172669101591106e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101591106e-06) [X2 X3 X11 X12]
+ (2.172669101591106e-06) [X2 Y3 Y11 X12]
+ (3.117447946529238e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946529238e-06) [X0 Z2 Z3 X4]
+ (3.5390541844008426e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844008426e-06) [X2 Z3 X4 Z12]
+ (3.5390541844008426e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844008426e-06) [X3 Z4 X5 Z13]
+ (4.281913884944483e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884944483e-06) [X4 Z5 X6 Z11]
+ (4.281913884944483e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884944483e-06) [X5 Z6 X7 Z10]
+ (5.275883122207715e-06) [Y3 X4 X12 Y13]
+ (5.275883122207715e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122207715e-06) [X3 X4 X12 X13]
+ (5.275883122207715e-06) [X3 Y4 Y12 X13]
+ (5.974311713465497e-06) [Y5 X6 X10 Y11]
+ (5.974311713465497e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713465497e-06) [X5 X6 X10 X11]
+ (5.974311713465497e-06) [X5 Y6 Y10 X11]
+ (7.954413176224722e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176224722e-06) [X10 Z11 X12 Z13]
+ (8.814937306608558e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306608558e-06) [X2 Z3 X4 Z13]
+ (8.814937306608558e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306608558e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110854) [Y7 X8 X9 Y10]
+ (0.0002921986261110854) [X7 Y8 Y9 X10]
+ (0.0004956762314916211) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916211) [X2 Z4 Z5 X6]
+ (0.0011059037691897012) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897012) [X0 Z1 X2 Z5]
+ (0.0011059037691897012) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897012) [X1 Z2 X3 Z4]
+ (0.0016638798784907743) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907743) [X2 Z3 Z4 X6]
+ (0.0016638798784907743) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907743) [X3 Z5 Z6 X7]
+ (0.0017560707018412659) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412659) [X0 Z1 X2 Z11]
+ (0.0017560707018412659) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412659) [X1 Z2 X3 Z10]
+ (0.002326230623158105) [Y0 Z1 Y2 Z13]
+ (0.002326230623158105) [X0 Z1 X2 Z13]
+ (0.002326230623158105) [Y1 Z2 Y3 Z12]
+ (0.002326230623158105) [X1 Z2 X3 Z12]
+ (0.0027458364701867977) [Y0 X1 X4 Y5]
+ (0.0027458364701867977) [X0 Y1 Y4 X5]
+ (0.0029297686747510885) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510885) [X0 Z1 X2 Z9]
+ (0.0029297686747510885) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510885) [X1 Z2 X3 Z8]
+ (0.003276971931231686) [Y0 Z1 Y2 Z3]
+ (0.003276971931231686) [X0 Z1 X2 Z3]
+ (0.003347617530666222) [Y0 Z1 Y2 Z7]
+ (0.003347617530666222) [X0 Z1 X2 Z7]
+ (0.003347617530666222) [Y1 Z2 Y3 Z6]
+ (0.003347617530666222) [X1 Z2 X3 Z6]
+ (0.003555290195504295) [Y0 Z1 Y2 Z10]
+ (0.003555290195504295) [X0 Z1 X2 Z10]
+ (0.003555290195504295) [Y1 Z2 Y3 Z11]
+ (0.003555290195504295) [X1 Z2 X3 Z11]
+ (0.005143391768825117) [Y3 Y4 X5 X6]
+ (0.005143391768825117) [X3 X4 Y5 Y6]
+ (0.005283776488402962) [Y0 X1 X12 Y13]
+ (0.005283776488402962) [X0 Y1 Y12 X13]
+ (0.005530759218631534) [Y0 Z1 Y2 Z4]
+ (0.005530759218631534) [X0 Z1 X2 Z4]
+ (0.005530759218631534) [Y1 Z2 Y3 Z5]
+ (0.005530759218631534) [X1 Z2 X3 Z5]
+ (0.006087822480561868) [Y8 X9 X12 Y13]
+ (0.006087822480561868) [X8 Y9 Y12 X13]
+ (0.0065093612011772346) [Y0 X1 X8 Y9]
+ (0.0065093612011772346) [X0 Y1 Y8 X9]
+ (0.0068881943529705775) [Y0 X1 X6 Y7]
+ (0.0068881943529705775) [X0 Y1 Y6 X7]
+ (0.006901238249797304) [Y0 Z1 Y2 Z12]
+ (0.006901238249797304) [X0 Z1 X2 Z12]
+ (0.006901238249797304) [Y1 Z2 Y3 Z13]
+ (0.006901238249797304) [X1 Z2 X3 Z13]
+ (0.007156934919856923) [Y4 X5 X8 Y9]
+ (0.007156934919856923) [X4 Y5 Y8 X9]
+ (0.007731425250775283) [Y0 X1 X10 Y11]
+ (0.007731425250775283) [X0 Y1 Y10 X11]
+ (0.008032520918821404) [Y0 Z1 Y2 Z6]
+ (0.008032520918821404) [X0 Z1 X2 Z6]
+ (0.008032520918821404) [Y1 Z2 Y3 Z7]
+ (0.008032520918821404) [X1 Z2 X3 Z7]
+ (0.009560705729135975) [Y8 X9 X10 Y11]
+ (0.009560705729135975) [X8 Y9 Y10 X11]
+ (0.0110550205961321) [Y0 Z1 Y2 Z8]
+ (0.0110550205961321) [X0 Z1 X2 Z8]
+ (0.0110550205961321) [Y1 Z2 Y3 Z9]
+ (0.0110550205961321) [X1 Z2 X3 Z9]
+ (0.011285190200840867) [Y5 Y6 X11 X12]
+ (0.011285190200840867) [X5 X6 Y11 Y12]
+ (0.01130727400884822) [Y7 Z8 Z9 Y11]
+ (0.01130727400884822) [X7 Z8 Z9 X11]
+ (0.01198238901024793) [Y4 X5 X6 Y7]
+ (0.01198238901024793) [X4 Y5 Y6 X7]
+ (0.013873381748426129) [Y6 X7 X8 Y9]
+ (0.013873381748426129) [X6 Y7 Y8 X9]
+ (0.01458364890761266) [Y0 X1 X2 Y3]
+ (0.01458364890761266) [X0 Y1 Y2 X3]
+ (0.015577208063976444) [Y2 X3 X12 Y13]
+ (0.015577208063976444) [X2 Y3 Y12 X13]
+ (0.017680067952481487) [Y4 X5 X10 Y11]
+ (0.017680067952481487) [X4 Y5 Y10 X11]
+ (0.01782514099578642) [Y6 X7 X10 Y11]
+ (0.01782514099578642) [X6 Y7 Y10 X11]
+ (0.01902824244384729) [Y3 X4 X11 Y12]
+ (0.01902824244384729) [X3 Y4 Y11 X12]
+ (0.02538465750845744) [Y2 X3 X10 Y11]
+ (0.02538465750845744) [X2 Y3 Y10 X11]
+ (0.028685183716105792) [Y10 X11 X12 Y13]
+ (0.028685183716105792) [X10 Y11 Y12 X13]
+ (0.029812424517345733) [Y6 Z7 Z8 Y10]
+ (0.029812424517345733) [X6 Z7 Z8 X10]
+ (0.029812424517345733) [Y7 Z9 Z10 Y11]
+ (0.029812424517345733) [X7 Z9 Z10 X11]
+ (0.030104623143456816) [Y6 Z7 Z9 Y10]
+ (0.030104623143456816) [X6 Z7 Z9 X10]
+ (0.030104623143456816) [Y7 Z8 Z10 Y11]
+ (0.030104623143456816) [X7 Z8 Z10 X11]
+ (0.030787505389143884) [Y6 Z8 Z9 Y10]
+ (0.030787505389143884) [X6 Z8 Z9 X10]
+ (0.031143817988967013) [Y2 X3 X6 Y7]
+ (0.031143817988967013) [X2 Y3 Y6 X7]
+ (0.03583956795335338) [Y2 X3 X4 Y5]
+ (0.03583956795335338) [X2 Y3 Y4 X5]
+ (0.03619412355904251) [Y2 X3 X8 Y9]
+ (0.03619412355904251) [X2 Y3 Y8 X9]
+ (0.038314670294803795) [Y4 X5 X12 Y13]
+ (0.038314670294803795) [X4 Y5 Y12 X13]
+ (0.10433064780651408) [Z0 Y1 Z2 Y3]
+ (0.10433064780651408) [Z0 X1 Z2 X3]
+ (-0.12133276911042416) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042416) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042411) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042411) [X3 Z4 Z5 Z6 X7]
+ (3.202076880230281e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880230281e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076880230282e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076880230282e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491888) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491888) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918882) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918882) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329059) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329059) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329059) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329059) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527321) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527321) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527321) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527321) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021263) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021263) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646218) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646218) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646218) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646218) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172953) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172953) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172953) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172953) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613896) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613896) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613896) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613896) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613896) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613896) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613896) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613896) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981928) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981928) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981928) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981928) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688777) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688777) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688777) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688777) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688777) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688777) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688777) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688777) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381013) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381013) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832961) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832961) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832961) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832961) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826939) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826939) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826939) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826939) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017372) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017372) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017372) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017372) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825117) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825117) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825117) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825117) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155181) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155181) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776298) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776298) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0045750076266392) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750076266392) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441833) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441833) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840064) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840064) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840064) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840064) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890217) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890217) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890217) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890217) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.00277902679902555) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.00277902679902555) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524715) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524715) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630292) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630292) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369458) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369458) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730291) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730291) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730291) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730291) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125482) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125482) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270957397) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270957397) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270957397) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270957397) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591773e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591773e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591773e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591773e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864557758e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864557758e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864557758e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864557758e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215684157e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215684157e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215684157e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215684157e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675850604e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675850604e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675850604e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675850604e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848499958e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848499958e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848499958e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848499958e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433039531e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433039531e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433039531e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433039531e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713465497e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713465497e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122207715e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122207715e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068428097e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068428097e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217994415e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217994415e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225616086e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225616086e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594517222295e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594517222295e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294408585e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294408585e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130323189e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130323189e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130323189e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130323189e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455002384897e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455002384897e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831952198462e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831952198462e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831952198462e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831952198462e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283482614683e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283482614683e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283482614683e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283482614683e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311123073e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311123073e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112289197e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112289197e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101591106e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101591106e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424492664056e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424492664056e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887071551e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887071551e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337826446263e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337826446263e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477603269385e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477603269385e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373858177e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373858177e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742071299e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742071299e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742071299e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742071299e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201769763e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201769763e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914663e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914663e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914663e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914663e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574691448e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574691448e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574691448e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574691448e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082573292e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082573292e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082573292e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082573292e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911514835e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911514835e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624757778e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624757778e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624757778e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624757778e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624757778e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624757778e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624757778e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624757778e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751512874e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751512874e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329674761e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329674761e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139351033427e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139351033427e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265654431823e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265654431823e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265654431823e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265654431823e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231291439617e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231291439617e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289482770074e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289482770074e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289482770074e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289482770074e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516185992589e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516185992589e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412773458046e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412773458046e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412773458046e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412773458046e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915756392e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915756392e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915756392e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915756392e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539177918198e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539177918198e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539177918198e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539177918198e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781482433854e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781482433854e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781482433854e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781482433854e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781482433854e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781482433854e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781482433854e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781482433854e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781482433854e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781482433854e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781482433854e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781482433854e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694865388208e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694865388208e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599406664e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599406664e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599406664e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599406664e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599406664e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599406664e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599406664e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599406664e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594980071e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594980071e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594980071e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594980071e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134754606e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134754606e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134754606e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134754606e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915756392e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915756392e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915756392e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915756392e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516185992589e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516185992589e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231291439617e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231291439617e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961752468e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961752468e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961752468e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961752468e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139351033427e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139351033427e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329674761e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329674761e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751512874e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751512874e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911514835e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911514835e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201769763e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201769763e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373858177e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373858177e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651982079e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651982079e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651982079e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651982079e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477603269385e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477603269385e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337826446263e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337826446263e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217425259e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217425259e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217425259e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217425259e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887071551e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887071551e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424492664056e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424492664056e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101591106e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101591106e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112289197e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112289197e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946529238e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946529238e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311123073e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311123073e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455002384897e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455002384897e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289383798e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289383798e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294408585e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294408585e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559541474e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559541474e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217994415e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217994415e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068428097e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068428097e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122207715e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122207715e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713465497e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713465497e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110854) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110854) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110854) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110854) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916211) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916211) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498458) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498458) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498458) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498458) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125482) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125482) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721375) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721375) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721375) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721375) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440573) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440573) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440573) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440573) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369458) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369458) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630292) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630292) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524715) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524715) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133923) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133923) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133923) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133923) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496529) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496529) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496529) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496529) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441833) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441833) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750076266392) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750076266392) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776298) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776298) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155181) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155181) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221641) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221641) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221641) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221641) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109477) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109477) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109477) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109477) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592156) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592156) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592156) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592156) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381013) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381013) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694593) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694593) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694593) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694593) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158512) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158512) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158512) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158512) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671565) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671565) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671565) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671565) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542516) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542516) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542516) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542516) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.01130727400884822) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.01130727400884822) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130827) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130827) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130827) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130827) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226567) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226567) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226567) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226567) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380154) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380154) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380154) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380154) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375477) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375477) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375477) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375477) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039907) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039907) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039907) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039907) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535457) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535457) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535457) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535457) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535457) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535457) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535457) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535457) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068935) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068935) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068935) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068935) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068935) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068935) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068935) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068935) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149384) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149384) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149384) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149384) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884452) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884452) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884452) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884452) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143884) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143884) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781298164) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781298164) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780755) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780755) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780755) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780755) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661346) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661346) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661346) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661346) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928377383e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928377383e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928377383e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928377383e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860067291834e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860067291834e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595086006729183e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086006729183e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013783124) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783124) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378314) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378314) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.047642612176383076) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383076) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383076) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383076) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982173) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982173) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982173) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982173) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289334) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289334) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289334) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289334) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205314) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205314) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205314) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205314) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197535) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197535) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197535) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197535) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312546) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312546) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624887) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624887) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624887) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624887) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190549) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190549) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190549) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190549) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026824) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026824) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026824) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026824) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891015) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891015) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891015) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891015) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693118) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693118) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952911) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952911) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601289) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601289) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600957) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600957) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600957) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600957) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01902824244384729) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384729) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494289) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494289) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494289) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494289) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179455) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179455) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226567) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226567) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162122) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162122) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172953) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172953) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981928) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981928) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840867) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840867) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962602) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962602) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847366) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847366) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847366) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847366) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102393) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102393) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832961) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832961) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561341) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561341) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017372) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017372) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109477) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109477) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840064) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840064) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832901) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832901) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832901) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832901) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235563) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235563) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235563) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235563) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255497) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255497) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066015) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066015) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066015) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066015) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352472) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352472) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352472) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352472) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696558) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696558) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696558) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696558) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696558) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696558) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696558) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696558) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569580997) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569580997) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549078) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549078) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549078) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549078) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591773e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591773e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305664516e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305664516e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305664516e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305664516e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879522354e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879522354e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879522354e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879522354e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775155362e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775155362e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775155362e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775155362e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994675801305e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994675801305e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994675801305e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994675801305e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209668929879e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209668929879e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209668929879e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209668929879e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833359085e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833359085e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833359085e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833359085e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736471123e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736471123e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736471123e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736471123e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220386842375e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220386842375e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220386842375e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220386842375e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147213406e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147213406e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147213406e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147213406e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225616086e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225616086e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594517222295e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594517222295e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954293047766e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954293047766e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954293047766e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954293047766e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954293047766e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954293047766e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954293047766e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954293047766e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320366724e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320366724e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320366724e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320366724e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045243946e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045243946e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045243946e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045243946e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122097931999e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122097931999e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122097931999e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122097931999e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836617505e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836617505e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836617505e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836617505e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117476928853e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117476928853e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117476928853e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117476928853e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.522493067600894e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.522493067600894e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.522493067600894e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.522493067600894e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.522493067600894e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067600894e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067600894e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.522493067600894e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337826446263e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337826446263e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337826446263e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337826446263e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770289636088e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770289636088e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770289636088e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770289636088e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104409747e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104409747e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104409747e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104409747e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975136815e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975136815e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207199961e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207199961e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744715338e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744715338e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471807585643e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471807585643e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471807585643e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471807585643e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896775978347e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896775978347e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108877525e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108877525e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108877525e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108877525e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139351033427e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139351033427e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139351033427e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139351033427e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565443182e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565443182e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293596886519e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596886519e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596886519e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293596886519e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289482770074e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289482770074e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209157563918e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209157563918e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594980071e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594980071e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096517378e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096517378e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096517378e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096517378e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594980071e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594980071e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350659239525e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350659239525e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350659239525e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350659239525e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355707944e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355707944e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355707944e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355707944e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209157563918e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209157563918e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289482770074e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289482770074e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565443182e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565443182e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896775978347e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896775978347e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744715338e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744715338e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207199961e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207199961e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975136815e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975136815e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887071551e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887071551e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887071551e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887071551e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435221788e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435221788e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435221788e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435221788e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514742905e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514742905e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514742905e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514742905e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400341167e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400341167e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400341167e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400341167e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400341167e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400341167e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400341167e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400341167e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420190751847e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190751847e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190751847e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190751847e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190751847e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190751847e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420190751847e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190751847e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455002384897e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455002384897e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455002384897e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455002384897e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289383798e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289383798e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559541474e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559541474e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591773e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591773e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569580997) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569580997) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840759) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840759) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840759) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840759) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005475) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005475) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005475) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005475) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005475) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005475) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005475) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005475) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125483) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125483) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125483) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125483) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907562) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907562) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907562) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907562) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496738) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496738) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496738) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496738) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.00130380047881269) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.00130380047881269) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.00130380047881269) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.00130380047881269) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482346) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482346) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482346) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482346) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482346) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482346) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482346) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482346) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619292) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619292) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619292) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619292) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840064) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840064) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914313) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914313) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914313) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914313) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182574) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182574) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182574) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182574) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660375) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660375) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660375) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660375) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660375) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660375) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660375) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660375) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803874) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803874) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803874) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803874) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076846) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076846) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076846) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076846) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109477) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109477) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0053799371558393635) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.0053799371558393635) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.0053799371558393635) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.0053799371558393635) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017372) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017372) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.0057084959859609215) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.0057084959859609215) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.0057084959859609215) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.0057084959859609215) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561341) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561341) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832961) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832961) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102393) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102393) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962602) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962602) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840867) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840867) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981928) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981928) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172953) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172953) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162122) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162122) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226567) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226567) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179455) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179455) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384729) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384729) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.045879470781298164) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298164) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156234) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156234) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156234) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156234) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702304) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702304) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767023027) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023027) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036467) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036467) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036467) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036467) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863616) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863616) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863616) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863616) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635018) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635018) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635018) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635018) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214034) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214034) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214034) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214034) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312546) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312546) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366165) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366165) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366165) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366165) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829916) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829916) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829916) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829916) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693118) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693118) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529113) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529113) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601289) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601289) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131478) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131478) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131478) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131478) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898928) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898928) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898928) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898928) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179455) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179455) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179455) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179455) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831728) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831728) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831728) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831728) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0098417492469626) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0098417492469626) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0098417492469626) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0098417492469626) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420984) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420984) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420984) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420984) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454837) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454837) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454837) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454837) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454837) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454837) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454837) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454837) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102393) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102393) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102393) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102393) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776298) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776298) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369533) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369533) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285325) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285325) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285325) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285325) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178704) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178704) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832901) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832901) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235563) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235563) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016236) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016236) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369458) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369458) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124228) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124228) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169506) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169506) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169506) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169506) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024449) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024449) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487642) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487642) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975606) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975606) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549078) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549078) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221158687e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221158687e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221158687e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221158687e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736471123e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736471123e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311123073e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311123073e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112289197e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112289197e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117065474257e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117065474257e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714216965e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714216965e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320366724e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320366724e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562131297e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562131297e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507247038e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507247038e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507247038e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507247038e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.35233210267935e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.35233210267935e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.35233210267935e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.35233210267935e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198763081e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198763081e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198763081e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198763081e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198763081e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198763081e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198763081e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198763081e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985615203e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985615203e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985615203e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985615203e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986081788e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986081788e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986081788e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986081788e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104409748e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104409748e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464560428e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464560428e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464560428e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464560428e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464560428e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464560428e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464560428e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464560428e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422030204e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422030204e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422030204e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422030204e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422030204e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422030204e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422030204e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422030204e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521165246e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521165246e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521165246e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521165246e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084839545e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393084839545e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084839545e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393084839545e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084839545e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393084839545e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393084839545e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393084839545e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293596886519e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293596886519e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815431811765e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815431811765e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355707944e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355707944e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350659239525e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350659239525e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242986553e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773242986553e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242986553e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773242986553e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242986553e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773242986553e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773242986553e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773242986553e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253789179866e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253789179866e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253789179866e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253789179866e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716553795589e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716553795589e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716553795589e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716553795589e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350659239525e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350659239525e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282181432873e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282181432873e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282181432873e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282181432873e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494530635e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494530635e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494530635e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494530635e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355707944e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355707944e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052280463e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052280463e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052280463e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052280463e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815431811765e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815431811765e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293596886519e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293596886519e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506160670646e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506160670646e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506160670646e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506160670646e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506160670646e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506160670646e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506160670646e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506160670646e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978543909146e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978543909146e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978543909146e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978543909146e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150952937167e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150952937167e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150952937167e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150952937167e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425343248e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425343248e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425343248e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425343248e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425343248e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425343248e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425343248e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425343248e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104409748e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104409748e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562131297e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562131297e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320366724e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320366724e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714216965e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714216965e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765762256087e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765762256087e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011696996e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011696996e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011696996e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011696996e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117065474257e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117065474257e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112289197e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112289197e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311123073e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311123073e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671383305e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671383305e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671383305e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671383305e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736471123e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736471123e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722163721e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722163721e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722163721e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722163721e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963275964345e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963275964345e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963275964345e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963275964345e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502191833e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502191833e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502191833e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502191833e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.42798865650995e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.42798865650995e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.42798865650995e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.42798865650995e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718244421e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718244421e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718244421e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718244421e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348296692e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348296692e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793585418e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793585418e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793585418e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793585418e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218875e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411218875e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411218875e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218875e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549078) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549078) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389547205) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389547205) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389547205) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389547205) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975606) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975606) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569580997) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580997) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569580997) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580997) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487642) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487642) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.000715673424890858) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.000715673424890858) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.000715673424890858) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000715673424890858) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024449) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024449) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730019) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730019) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730019) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730019) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124228) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124228) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369458) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369458) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158505) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158505) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158505) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158505) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235563) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235563) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832901) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832901) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178704) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178704) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369533) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369533) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776298) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776298) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278092) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278092) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278092) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278092) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226856) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226856) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226856) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226856) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409955) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409955) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409955) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409955) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561341) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561341) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561341) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561341) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796737) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796737) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796737) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796737) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908924) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908924) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908924) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908924) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162122) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162122) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162122) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162122) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363766) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363766) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363766) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363766) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363766) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363766) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363766) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363766) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386176) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386176) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505274317734e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505274317734e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505274317734e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505274317734e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002437) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002437) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002438) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002438) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.010311482489831728) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831728) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420984) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420984) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297706035) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297706035) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297706035) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297706035) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311857) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311857) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311857) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311857) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311857) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311857) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311857) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311857) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766165) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766165) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766165) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766165) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728533) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728533) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219316) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219316) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219316) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219316) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158505) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158505) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939865) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939865) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939865) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939865) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016236) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016236) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587448) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587448) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587448) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587448) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587448) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587448) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587448) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587448) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124226) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124226) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124226) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124226) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538392) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538392) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538392) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538392) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538392) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538392) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538392) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538392) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562784) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562784) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562784) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562784) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453184855e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453184855e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714216965e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714216965e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714216965e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714216965e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562131297e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562131297e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562131297e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562131297e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129831166e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129831166e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129831166e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129831166e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230214273e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230214273e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230214273e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230214273e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037218883e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037218883e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037218883e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037218883e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.66134721327391e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.66134721327391e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.66134721327391e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.66134721327391e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413745052e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413745052e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975136815e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975136815e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658424391e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658424391e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658424391e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658424391e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207199961e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207199961e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896775978347e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896775978347e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325323515027e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325323515027e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325323515027e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325323515027e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458943895e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458943895e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998845666096e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998845666096e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998845666096e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998845666096e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731755278914e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731755278914e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731755278914e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731755278914e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929953896e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641929953896e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315636987e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315636987e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315636987e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315636987e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641929953896e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641929953896e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815431811765e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815431811765e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815431811765e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815431811765e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458943895e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458943895e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896775978347e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896775978347e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390507594e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390507594e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390507594e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390507594e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207199961e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207199961e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975136815e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975136815e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413745052e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413745052e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487205302e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487205302e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577218432e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577218432e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577218432e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577218432e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765762256082e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765762256082e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117065474257e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117065474257e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117065474257e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117065474257e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348296691e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348296691e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735543253e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735543253e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735543253e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735543253e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693265093e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693265093e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693265093e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693265093e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487643) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487643) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487643) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487643) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024449) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024449) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024449) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024449) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441826) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441826) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441826) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441826) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245112) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245112) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245112) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245112) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500461) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500461) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500461) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500461) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798022) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798022) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798022) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798022) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798022) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798022) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798022) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798022) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158505) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158505) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728533) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728533) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369533) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369533) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369533) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369533) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046444) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046444) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046444) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046444) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420984) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420984) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831728) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831728) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386176) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386176) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009016112237e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009016112237e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009016112237e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009016112237e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178704) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178704) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121932) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121932) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975606) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975606) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453184855e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453184855e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577218432e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577218432e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413745052e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413745052e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413745052e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413745052e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641929953896e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929953896e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929953896e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929953896e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458943895e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458943895e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458943895e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458943895e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487205303e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487205303e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577218432e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577218432e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975606) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975606) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121932) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121932) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178704) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178704) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XYI =  0.022659767960222288
Expectation value of XIZ =  0.07715357869738898
[0.27361669 0.00898685 0.26297431 0.00732554 0.21720814 0.00116213
 0.22790267 0.00082366]
Expectation value of XYI =  0.022659767960222343
Expectation value of XIZ =  0.07715357869738915
[0.02265977 0.07715358]
[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[1])]
[PauliZ(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[0]) @ PauliZ(wires=[2])]
pennylane.qnodes.base.QuantumFunctionError: Only observables that are qubit-wise commuting
Pauli words can be returned on the same wire
Minimum number of QWC groupings found: 2
Group 0:
Y0 X2 X3
Y0 Y1 X2 X3
X2 X3
Group 1:
Z0 Z1 Z2
Z0 Z1 Z2 Z3
Z0
Z0 Z1
Term expectation values:
Group 0 expectation values: [-0.14012997  0.01555488  0.18967764]
Group 1 expectation values: [0.93755207 0.94996042 0.96302938 0.96118149]
<H> =  3.8768259168631207
3.8768259168631207
Number of Hamiltonian terms/required measurements: 2050
Number of required measurements after optimization: 523
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
   (-46.46390678868893) [I0]
+ (0.7829661725950189) [Z10]
+ (0.782966172595019) [Z11]
+ (0.8084581961720474) [Z12]
+ (0.8084581961720475) [Z13]
+ (1.2034402289145634) [Z4]
+ (1.2034402289145634) [Z5]
+ (1.3096862988615425) [Z6]
+ (1.3096862988615428) [Z7]
+ (1.3693525634718187) [Z8]
+ (1.3693525634718189) [Z9]
+ (1.6538942226831712) [Z3]
+ (1.6538942226831714) [Z2]
+ (-8.194261372148781e-06) [Y10 Y12]
+ (-8.194261372148781e-06) [X10 X12]
+ (-1.854060857923412e-06) [Y5 Y7]
+ (-1.854060857923412e-06) [X5 X7]
+ (-7.764994120107066e-07) [Y3 Y5]
+ (-7.764994120107066e-07) [X3 X5]
+ (-5.929765814709979e-07) [Y4 Y6]
+ (-5.929765814709979e-07) [X4 X6]
+ (1.602116740790288e-06) [Y2 Y4]
+ (1.602116740790288e-06) [X2 X4]
+ (7.954413176340711e-06) [Y11 Y13]
+ (7.954413176340711e-06) [X11 X13]
+ (0.0032769719312316266) [Y1 Y3]
+ (0.0032769719312316266) [X1 X3]
+ (0.11270386920332208) [Z10 Z12]
+ (0.11270386920332208) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.11952438964682673) [Z6 Z10]
+ (0.11952438964682673) [Z7 Z11]
+ (0.12489990917237592) [Z4 Z10]
+ (0.12489990917237592) [Z5 Z11]
+ (0.12495807739503223) [Z2 Z4]
+ (0.12495807739503223) [Z3 Z5]
+ (0.12799502492468412) [Z2 Z10]
+ (0.12799502492468412) [Z3 Z11]
+ (0.134017152619637) [Z6 Z12]
+ (0.134017152619637) [Z7 Z13]
+ (0.13701191674040758) [Z4 Z6]
+ (0.13701191674040758) [Z5 Z7]
+ (0.1373495306426133) [Z6 Z11]
+ (0.1373495306426133) [Z7 Z10]
+ (0.13739104762683235) [Z2 Z6]
+ (0.13739104762683235) [Z3 Z7]
+ (0.13766872645852565) [Z8 Z10]
+ (0.13766872645852565) [Z9 Z11]
+ (0.14011289865354815) [Z2 Z12]
+ (0.14011289865354815) [Z3 Z13]
+ (0.14138905291942805) [Z10 Z13]
+ (0.14138905291942805) [Z11 Z12]
+ (0.14257997712485743) [Z4 Z11]
+ (0.14257997712485743) [Z5 Z10]
+ (0.14722943218766155) [Z8 Z11]
+ (0.14722943218766155) [Z9 Z10]
+ (0.14899430575065553) [Z4 Z7]
+ (0.14899430575065553) [Z5 Z6]
+ (0.1492635514738889) [Z10 Z11]
+ (0.14960702684445296) [Z4 Z8]
+ (0.14960702684445296) [Z5 Z9]
+ (0.14973486803496916) [Z8 Z12]
+ (0.14973486803496916) [Z9 Z13]
+ (0.15071408121008287) [Z2 Z8]
+ (0.15071408121008287) [Z3 Z9]
+ (0.15138327161428844) [Z6 Z13]
+ (0.15138327161428844) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314143) [Z2 Z11]
+ (0.15337968243314143) [Z3 Z10]
+ (0.15435748657223636) [Z12 Z13]
+ (0.1556901067175246) [Z2 Z13]
+ (0.1556901067175246) [Z3 Z12]
+ (0.155822690515531) [Z8 Z13]
+ (0.155822690515531) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985662) [Z4 Z5]
+ (0.16079764534838564) [Z2 Z5]
+ (0.16079764534838564) [Z3 Z4]
+ (0.1675665326546127) [Z6 Z8]
+ (0.1675665326546127) [Z7 Z9]
+ (0.16853486561579953) [Z2 Z7]
+ (0.16853486561579953) [Z3 Z6]
+ (0.18143991440303875) [Z6 Z9]
+ (0.18143991440303875) [Z7 Z8]
+ (0.18189085790751372) [Z2 Z3]
+ (0.18690820476912554) [Z2 Z9]
+ (0.18690820476912554) [Z3 Z8]
+ (0.19299723935364219) [Z0 Z10]
+ (0.19299723935364219) [Z1 Z11]
+ (0.19392534613270204) [Z6 Z7]
+ (0.1966177089034215) [Z0 Z4]
+ (0.1966177089034215) [Z1 Z5]
+ (0.19936354537360834) [Z0 Z5]
+ (0.19936354537360834) [Z1 Z4]
+ (0.2007286646044174) [Z0 Z11]
+ (0.2007286646044174) [Z1 Z10]
+ (0.21102659849791494) [Z0 Z12]
+ (0.21102659849791494) [Z1 Z13]
+ (0.2163103749863179) [Z0 Z13]
+ (0.2163103749863179) [Z1 Z12]
+ (0.2200397733437608) [Z8 Z9]
+ (0.23671080783830437) [Z0 Z2]
+ (0.23671080783830437) [Z1 Z3]
+ (0.24164663936017197) [Z0 Z6]
+ (0.24164663936017197) [Z1 Z7]
+ (0.25129445674591705) [Z0 Z3]
+ (0.25129445674591705) [Z1 Z2]
+ (0.2723251830660566) [Z0 Z8]
+ (0.2723251830660566) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860489) [Z0 Z1]
+ (-1.2260484988579439e-05) [Y5 Z6 Y7]
+ (-1.2260484988579439e-05) [X5 Z6 X7]
+ (-1.2260484988579436e-05) [Y4 Z5 Y6]
+ (-1.2260484988579436e-05) [X4 Z5 X6]
+ (-1.0722312157312484e-05) [Y10 Z11 Y12]
+ (-1.0722312157312484e-05) [X10 Z11 X12]
+ (-1.0722312157312484e-05) [Y11 Z12 Y13]
+ (-1.0722312157312484e-05) [X11 Z12 X13]
+ (-3.887051673759952e-06) [Y2 Z3 Y4]
+ (-3.887051673759952e-06) [X2 Z3 X4]
+ (-3.887051673759952e-06) [Y3 Z4 Y5]
+ (-3.887051673759952e-06) [X3 Z4 X5]
+ (0.12507032579771957) [Y0 Z1 Y2]
+ (0.12507032579771957) [X0 Z1 X2]
+ (0.12507032579771957) [Y1 Z2 Y3]
+ (0.12507032579771957) [X1 Z2 X3]
+ (-0.03831467029480388) [Y4 Y5 X12 X13]
+ (-0.03831467029480388) [X4 X5 Y12 Y13]
+ (-0.03619412355904266) [Y2 Y3 X8 X9]
+ (-0.03619412355904266) [X2 X3 Y8 Y9]
+ (-0.03583956795335342) [Y2 Y3 X4 X5]
+ (-0.03583956795335342) [X2 X3 Y4 Y5]
+ (-0.031143817988967176) [Y2 Y3 X6 X7]
+ (-0.031143817988967176) [X2 X3 Y6 Y7]
+ (-0.02868518371610599) [Y10 Y11 X12 X13]
+ (-0.02868518371610599) [X10 X11 Y12 Y13]
+ (-0.025996177598021086) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021086) [X3 Z4 Z5 X7]
+ (-0.02538465750845733) [Y2 Y3 X10 X11]
+ (-0.02538465750845733) [X2 X3 Y10 Y11]
+ (-0.019028242443847203) [Y3 Y4 X11 X12]
+ (-0.019028242443847203) [X3 X4 Y11 Y12]
+ (-0.01782514099578655) [Y6 Y7 X10 X11]
+ (-0.01782514099578655) [X6 X7 Y10 Y11]
+ (-0.017680067952481494) [Y4 Y5 X10 X11]
+ (-0.017680067952481494) [X4 X5 Y10 Y11]
+ (-0.015577208063976432) [Y2 Y3 X12 X13]
+ (-0.015577208063976432) [X2 X3 Y12 Y13]
+ (-0.014583648907612696) [Y0 Y1 X2 X3]
+ (-0.014583648907612696) [X0 X1 Y2 Y3]
+ (-0.013873381748426072) [Y6 Y7 X8 X9]
+ (-0.013873381748426072) [X6 X7 Y8 Y9]
+ (-0.011982389010247969) [Y4 Y5 X6 X7]
+ (-0.011982389010247969) [X4 X5 Y6 Y7]
+ (-0.011285190200840945) [Y5 X6 X11 Y12]
+ (-0.011285190200840945) [X5 Y6 Y11 X12]
+ (-0.009560705729135898) [Y8 Y9 X10 X11]
+ (-0.009560705729135898) [X8 X9 Y10 Y11]
+ (-0.008125251921381043) [Y1 X2 X8 Y9]
+ (-0.008125251921381043) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381043) [X1 X2 X8 X9]
+ (-0.008125251921381043) [X1 Y2 Y8 X9]
+ (-0.007731425250775254) [Y0 Y1 X10 X11]
+ (-0.007731425250775254) [X0 X1 Y10 Y11]
+ (-0.0071569349198569564) [Y4 Y5 X8 X9]
+ (-0.0071569349198569564) [X4 X5 Y8 Y9]
+ (-0.006888194352970554) [Y0 Y1 X6 X7]
+ (-0.006888194352970554) [X0 X1 Y6 Y7]
+ (-0.006509361201177239) [Y0 Y1 X8 X9]
+ (-0.006509361201177239) [X0 X1 Y8 Y9]
+ (-0.00608782248056184) [Y8 Y9 X12 X13]
+ (-0.00608782248056184) [X8 X9 Y12 Y13]
+ (-0.005283776488402953) [Y0 Y1 X12 X13]
+ (-0.005283776488402953) [X0 X1 Y12 Y13]
+ (-0.005143391768825098) [Y3 X4 X5 Y6]
+ (-0.005143391768825098) [X3 Y4 Y5 X6]
+ (-0.004684903388155222) [Y1 X2 X6 Y7]
+ (-0.004684903388155222) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155222) [X1 X2 X6 X7]
+ (-0.004684903388155222) [X1 Y2 Y6 X7]
+ (-0.004575007626639202) [Y1 X2 X12 Y13]
+ (-0.004575007626639202) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639202) [X1 X2 X12 X13]
+ (-0.004575007626639202) [X1 Y2 Y12 X13]
+ (-0.0044248554494418675) [Y1 X2 X4 Y5]
+ (-0.0044248554494418675) [Y1 Y2 Y4 Y5]
+ (-0.0044248554494418675) [X1 X2 X4 X5]
+ (-0.0044248554494418675) [X1 Y2 Y4 X5]
+ (-0.0034795118903343247) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343247) [X2 Z3 Z5 X6]
+ (-0.0034795118903343247) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343247) [X3 Z4 Z6 X7]
+ (-0.0027458364701868207) [Y0 Y1 X4 X5]
+ (-0.0027458364701868207) [X0 X1 Y4 Y5]
+ (-0.0017992194936630338) [Y1 X2 X10 Y11]
+ (-0.0017992194936630338) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630338) [X1 X2 X10 X11]
+ (-0.0017992194936630338) [X1 Y2 Y10 X11]
+ (-0.00029219862611102004) [Y7 Y8 X9 X10]
+ (-0.00029219862611102004) [X7 X8 Y9 Y10]
+ (-8.194261372148781e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372148781e-06) [Z10 X11 Z12 X13]
+ (-7.801707500518195e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500518195e-06) [X2 Z3 X4 Z11]
+ (-7.801707500518195e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500518195e-06) [X3 Z4 X5 Z10]
+ (-4.643051068509867e-06) [Y3 X4 X10 Y11]
+ (-4.643051068509867e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068509867e-06) [X3 X4 X10 X11]
+ (-4.643051068509867e-06) [X3 Y4 Y10 X11]
+ (-4.588855155640162e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155640162e-06) [X4 Z5 X6 Z13]
+ (-4.588855155640162e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155640162e-06) [X5 Z6 X7 Z12]
+ (-4.5565692181757895e-06) [Y5 X6 X12 Y13]
+ (-4.5565692181757895e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692181757895e-06) [X5 X6 X12 X13]
+ (-4.5565692181757895e-06) [X5 Y6 Y12 X13]
+ (-3.694513294545135e-06) [Y4 X5 X11 Y12]
+ (-3.694513294545135e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294545135e-06) [X4 X5 X11 X12]
+ (-3.694513294545135e-06) [X4 Y5 Y11 X12]
+ (-3.3440815564546245e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815564546245e-06) [Z0 X5 Z6 X7]
+ (-3.3440815564546245e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815564546245e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320083283e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320083283e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320083283e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320083283e-06) [X3 Z4 X5 Z11]
+ (-3.099349243565418e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243565418e-06) [Z0 X4 Z5 X6]
+ (-3.099349243565418e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243565418e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816231484e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816231484e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816231484e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816231484e-06) [Z7 X10 Z11 X12]
+ (-2.177664604931923e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604931923e-06) [Z0 X10 Z11 X12]
+ (-2.177664604931923e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604931923e-06) [Z1 X11 Z12 X13]
+ (-1.881850183168244e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183168244e-06) [X4 Z5 X6 Z9]
+ (-1.881850183168244e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183168244e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214863482e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214863482e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214863482e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214863482e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579234122e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579234122e-06) [X4 Z5 X6 Z7]
+ (-1.8163031698071153e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031698071153e-06) [Z4 X11 Z12 X13]
+ (-1.8163031698071153e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031698071153e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285240684e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285240684e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285240684e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285240684e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137720531e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137720531e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137720531e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137720531e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977542606e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977542606e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977542606e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977542606e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490841364e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490841364e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490841364e-06) [X3 X4 X6 X7]
+ (-1.4548424490841364e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080606118e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080606118e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080606118e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080606118e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099558723e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099558723e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099558723e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099558723e-06) [X3 Z4 X5 Z6]
+ (-1.190850808390296e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808390296e-06) [Z0 X3 Z4 X5]
+ (-1.190850808390296e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808390296e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369625686e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369625686e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369625686e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369625686e-06) [Z3 X4 Z5 X6]
+ (-1.0632283422652832e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283422652832e-06) [Z2 X10 Z11 X12]
+ (-1.0632283422652832e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283422652832e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601368e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601368e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601368e-06) [X6 X7 X11 X12]
+ (-1.0358477601368e-06) [X6 Y7 Y11 X12]
+ (-9.509249751252545e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751252545e-07) [Z2 X4 Z5 X6]
+ (-9.509249751252545e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751252545e-07) [Z3 X5 Z6 X7]
+ (-9.344557775878556e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775878556e-07) [Z8 X11 Z12 X13]
+ (-9.344557775878556e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775878556e-07) [Z9 X10 Z11 X12]
+ (-8.337746754598686e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754598686e-07) [Z0 X2 Z3 X4]
+ (-8.337746754598686e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754598686e-07) [Z1 X3 Z4 X5]
+ (-7.9568953730526e-07) [Y3 X4 X8 Y9]
+ (-7.9568953730526e-07) [Y3 Y4 Y8 Y9]
+ (-7.9568953730526e-07) [X3 X4 X8 X9]
+ (-7.9568953730526e-07) [X3 Y4 Y8 X9]
+ (-7.764994120107065e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120107065e-07) [X2 Z3 X4 Z5]
+ (-5.929765814709979e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814709979e-07) [Z4 X5 Z6 X7]
+ (-5.770052995175474e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995175474e-07) [X2 Z3 X4 Z9]
+ (-5.770052995175474e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995175474e-07) [X3 Z4 X5 Z8]
+ (-5.471647744697864e-07) [Y1 Y2 X11 X12]
+ (-5.471647744697864e-07) [X1 X2 Y11 Y12]
+ (-4.838052751076324e-07) [Y5 X6 X8 Y9]
+ (-4.838052751076324e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751076324e-07) [X5 X6 X8 X9]
+ (-4.838052751076324e-07) [X5 Y6 Y8 X9]
+ (-3.570761329304274e-07) [Y0 X1 X3 Y4]
+ (-3.570761329304274e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329304274e-07) [X0 X1 X3 X4]
+ (-3.570761329304274e-07) [X0 Y1 Y3 X4]
+ (-2.4473231288920663e-07) [Y0 X1 X5 Y6]
+ (-2.4473231288920663e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231288920663e-07) [X0 X1 X5 X6]
+ (-2.4473231288920663e-07) [X0 Y1 Y5 X6]
+ (-2.199051618373139e-07) [Y2 X3 X5 Y6]
+ (-2.199051618373139e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618373139e-07) [X2 X3 X5 X6]
+ (-2.199051618373139e-07) [X2 Y3 Y5 X6]
+ (-1.9332412772575493e-07) [Y1 X2 X3 Y4]
+ (-1.9332412772575493e-07) [X1 Y2 Y3 X4]
+ (-1.2919694863976733e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694863976733e-07) [X1 Z2 Z3 X5]
+ (1.7379332625511162e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332625511162e-07) [X0 Z1 Z3 X4]
+ (1.7379332625511162e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332625511162e-07) [X1 Z2 Z4 X5]
+ (1.9332412772575493e-07) [Y1 Y2 X3 X4]
+ (1.9332412772575493e-07) [X1 X2 Y3 Y4]
+ (2.1868423778771268e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423778771268e-07) [X2 Z3 X4 Z8]
+ (2.1868423778771268e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423778771268e-07) [X3 Z4 X5 Z9]
+ (2.5935343912826396e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343912826396e-07) [X2 Z3 X4 Z6]
+ (2.5935343912826396e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343912826396e-07) [X3 Z4 X5 Z7]
+ (3.6060718682397396e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718682397396e-07) [X0 Z1 Z2 X4]
+ (3.6060718682397396e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718682397396e-07) [X1 Z3 Z4 X5]
+ (5.471647744697864e-07) [Y1 X2 X11 Y12]
+ (5.471647744697864e-07) [X1 Y2 Y11 X12]
+ (5.627851911598693e-07) [Y0 X1 X11 Y12]
+ (5.627851911598693e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911598693e-07) [X0 X1 X11 X12]
+ (5.627851911598693e-07) [X0 Y1 Y11 X12]
+ (6.628614201664051e-07) [Y8 X9 X11 Y12]
+ (6.628614201664051e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201664051e-07) [X8 X9 X11 X12]
+ (6.628614201664051e-07) [X8 Y9 Y11 X12]
+ (1.1094407593068985e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407593068985e-06) [Z2 X11 Z12 X13]
+ (1.1094407593068985e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407593068985e-06) [Z3 X10 Z11 X12]
+ (1.602116740790288e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740790288e-06) [Z2 X3 Z4 X5]
+ (1.8782101247380191e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247380191e-06) [Z4 X10 Z11 X12]
+ (1.8782101247380191e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247380191e-06) [Z5 X11 Z12 X13]
+ (2.1726691015721818e-06) [Y2 X3 X11 Y12]
+ (2.1726691015721818e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015721818e-06) [X2 X3 X11 X12]
+ (2.1726691015721818e-06) [X2 Y3 Y11 X12]
+ (3.117447946300534e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946300534e-06) [X0 Z2 Z3 X4]
+ (3.5390541844712688e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844712688e-06) [X2 Z3 X4 Z12]
+ (3.5390541844712688e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844712688e-06) [X3 Z4 X5 Z13]
+ (4.281913884945513e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884945513e-06) [X4 Z5 X6 Z11]
+ (4.281913884945513e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884945513e-06) [X5 Z6 X7 Z10]
+ (5.275883122202624e-06) [Y3 X4 X12 Y13]
+ (5.275883122202624e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122202624e-06) [X3 X4 X12 X13]
+ (5.275883122202624e-06) [X3 Y4 Y12 X13]
+ (5.974311713469582e-06) [Y5 X6 X10 Y11]
+ (5.974311713469582e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713469582e-06) [X5 X6 X10 X11]
+ (5.974311713469582e-06) [X5 Y6 Y10 X11]
+ (7.954413176340711e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176340711e-06) [X10 Z11 X12 Z13]
+ (8.814937306673893e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306673893e-06) [X2 Z3 X4 Z13]
+ (8.814937306673893e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306673893e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611102004) [Y7 X8 X9 Y10]
+ (0.00029219862611102004) [X7 Y8 Y9 X10]
+ (0.0004956762314916396) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916396) [X2 Z4 Z5 X6]
+ (0.0011059037691896578) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896578) [X0 Z1 X2 Z5]
+ (0.0011059037691896578) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896578) [X1 Z2 X3 Z4]
+ (0.0016638798784907737) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907737) [X2 Z3 Z4 X6]
+ (0.0016638798784907737) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907737) [X3 Z5 Z6 X7]
+ (0.00175607070184121) [Y0 Z1 Y2 Z11]
+ (0.00175607070184121) [X0 Z1 X2 Z11]
+ (0.00175607070184121) [Y1 Z2 Y3 Z10]
+ (0.00175607070184121) [X1 Z2 X3 Z10]
+ (0.0023262306231580532) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580532) [X0 Z1 X2 Z13]
+ (0.0023262306231580532) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580532) [X1 Z2 X3 Z12]
+ (0.0027458364701868207) [Y0 X1 X4 Y5]
+ (0.0027458364701868207) [X0 Y1 Y4 X5]
+ (0.0029297686747510143) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510143) [X0 Z1 X2 Z9]
+ (0.0029297686747510143) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510143) [X1 Z2 X3 Z8]
+ (0.003276971931231627) [Y0 Z1 Y2 Z3]
+ (0.003276971931231627) [X0 Z1 X2 Z3]
+ (0.003347617530666148) [Y0 Z1 Y2 Z7]
+ (0.003347617530666148) [X0 Z1 X2 Z7]
+ (0.003347617530666148) [Y1 Z2 Y3 Z6]
+ (0.003347617530666148) [X1 Z2 X3 Z6]
+ (0.003555290195504244) [Y0 Z1 Y2 Z10]
+ (0.003555290195504244) [X0 Z1 X2 Z10]
+ (0.003555290195504244) [Y1 Z2 Y3 Z11]
+ (0.003555290195504244) [X1 Z2 X3 Z11]
+ (0.005143391768825098) [Y3 Y4 X5 X6]
+ (0.005143391768825098) [X3 X4 Y5 Y6]
+ (0.005283776488402953) [Y0 X1 X12 Y13]
+ (0.005283776488402953) [X0 Y1 Y12 X13]
+ (0.005530759218631526) [Y0 Z1 Y2 Z4]
+ (0.005530759218631526) [X0 Z1 X2 Z4]
+ (0.005530759218631526) [Y1 Z2 Y3 Z5]
+ (0.005530759218631526) [X1 Z2 X3 Z5]
+ (0.00608782248056184) [Y8 X9 X12 Y13]
+ (0.00608782248056184) [X8 Y9 Y12 X13]
+ (0.006509361201177239) [Y0 X1 X8 Y9]
+ (0.006509361201177239) [X0 Y1 Y8 X9]
+ (0.006888194352970554) [Y0 X1 X6 Y7]
+ (0.006888194352970554) [X0 Y1 Y6 X7]
+ (0.0069012382497972554) [Y0 Z1 Y2 Z12]
+ (0.0069012382497972554) [X0 Z1 X2 Z12]
+ (0.0069012382497972554) [Y1 Z2 Y3 Z13]
+ (0.0069012382497972554) [X1 Z2 X3 Z13]
+ (0.0071569349198569564) [Y4 X5 X8 Y9]
+ (0.0071569349198569564) [X4 Y5 Y8 X9]
+ (0.007731425250775254) [Y0 X1 X10 Y11]
+ (0.007731425250775254) [X0 Y1 Y10 X11]
+ (0.008032520918821371) [Y0 Z1 Y2 Z6]
+ (0.008032520918821371) [X0 Z1 X2 Z6]
+ (0.008032520918821371) [Y1 Z2 Y3 Z7]
+ (0.008032520918821371) [X1 Z2 X3 Z7]
+ (0.009560705729135898) [Y8 X9 X10 Y11]
+ (0.009560705729135898) [X8 Y9 Y10 X11]
+ (0.011055020596132056) [Y0 Z1 Y2 Z8]
+ (0.011055020596132056) [X0 Z1 X2 Z8]
+ (0.011055020596132056) [Y1 Z2 Y3 Z9]
+ (0.011055020596132056) [X1 Z2 X3 Z9]
+ (0.011285190200840945) [Y5 Y6 X11 X12]
+ (0.011285190200840945) [X5 X6 Y11 Y12]
+ (0.011307274008848124) [Y7 Z8 Z9 Y11]
+ (0.011307274008848124) [X7 Z8 Z9 X11]
+ (0.011982389010247969) [Y4 X5 X6 Y7]
+ (0.011982389010247969) [X4 Y5 Y6 X7]
+ (0.013873381748426072) [Y6 X7 X8 Y9]
+ (0.013873381748426072) [X6 Y7 Y8 X9]
+ (0.014583648907612696) [Y0 X1 X2 Y3]
+ (0.014583648907612696) [X0 Y1 Y2 X3]
+ (0.015577208063976432) [Y2 X3 X12 Y13]
+ (0.015577208063976432) [X2 Y3 Y12 X13]
+ (0.017680067952481494) [Y4 X5 X10 Y11]
+ (0.017680067952481494) [X4 Y5 Y10 X11]
+ (0.01782514099578655) [Y6 X7 X10 Y11]
+ (0.01782514099578655) [X6 Y7 Y10 X11]
+ (0.019028242443847203) [Y3 X4 X11 Y12]
+ (0.019028242443847203) [X3 Y4 Y11 X12]
+ (0.02538465750845733) [Y2 X3 X10 Y11]
+ (0.02538465750845733) [X2 Y3 Y10 X11]
+ (0.02868518371610599) [Y10 X11 X12 Y13]
+ (0.02868518371610599) [X10 Y11 Y12 X13]
+ (0.029812424517345844) [Y6 Z7 Z8 Y10]
+ (0.029812424517345844) [X6 Z7 Z8 X10]
+ (0.029812424517345844) [Y7 Z9 Z10 Y11]
+ (0.029812424517345844) [X7 Z9 Z10 X11]
+ (0.03010462314345686) [Y6 Z7 Z9 Y10]
+ (0.03010462314345686) [X6 Z7 Z9 X10]
+ (0.03010462314345686) [Y7 Z8 Z10 Y11]
+ (0.03010462314345686) [X7 Z8 Z10 X11]
+ (0.03078750538914398) [Y6 Z8 Z9 Y10]
+ (0.03078750538914398) [X6 Z8 Z9 X10]
+ (0.031143817988967176) [Y2 X3 X6 Y7]
+ (0.031143817988967176) [X2 Y3 Y6 X7]
+ (0.03583956795335342) [Y2 X3 X4 Y5]
+ (0.03583956795335342) [X2 Y3 Y4 X5]
+ (0.03619412355904266) [Y2 X3 X8 Y9]
+ (0.03619412355904266) [X2 Y3 Y8 X9]
+ (0.03831467029480388) [Y4 X5 X12 Y13]
+ (0.03831467029480388) [X4 Y5 Y12 X13]
+ (0.10433064780651406) [Z0 Y1 Z2 Y3]
+ (0.10433064780651406) [Z0 X1 Z2 X3]
+ (-0.12133276911042248) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042248) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042247) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042247) [X3 Z4 Z5 Z6 X7]
+ (3.2020768805374413e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768805374413e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768805374418e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768805374418e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918852) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918852) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918852) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918852) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329038) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329038) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329038) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329038) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273048) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273048) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273048) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273048) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802109) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802109) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646093) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646093) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646093) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646093) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173048) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173048) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173048) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173048) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613983) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613983) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613983) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613983) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613983) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613983) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613983) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613983) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819217) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819217) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819217) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819217) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688723) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688723) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688723) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688723) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688723) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688723) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688723) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688723) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381043) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381043) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832954) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832954) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832954) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832954) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826878) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826878) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826878) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826878) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017334) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017334) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017334) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017334) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825098) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825098) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825098) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825098) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155223) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155223) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.0046686203187763006) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.0046686203187763006) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639202) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639202) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.0044248554494418675) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.0044248554494418675) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840032) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840032) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840032) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840032) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.00349379035989005) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.00349379035989005) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.00349379035989005) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.00349379035989005) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255415) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255415) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524654) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524654) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630338) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630338) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369555) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369555) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730389) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730389) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730389) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730389) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125476) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125476) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956278) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956278) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956278) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956278) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880588069e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880588069e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880588069e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880588069e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.77481786468791e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77481786468791e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77481786468791e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77481786468791e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215755502e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215755502e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215755502e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215755502e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675987327e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675987327e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675987327e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675987327e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848576798e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848576798e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848576798e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848576798e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433262661e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433262661e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433262661e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433262661e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713469582e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713469582e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122202625e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122202625e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068509867e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068509867e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692181757895e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692181757895e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225572881e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225572881e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594520054206e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594520054206e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294545135e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294545135e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971305931195e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971305931195e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971305931195e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971305931195e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500156619e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500156619e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831955383615e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831955383615e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831955383615e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831955383615e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348420178e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348420178e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348420178e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348420178e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.15134631114171e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.15134631114171e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507113275385e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507113275385e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015721818e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015721818e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490841362e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490841362e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887005826e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887005826e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824928426e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824928426e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601368e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601368e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.9568953730526e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.9568953730526e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742248266e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742248266e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742248266e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742248266e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201664051e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201664051e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914676989e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914676989e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914676989e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914676989e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574748486e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574748486e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574748486e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574748486e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.92745308278754e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.92745308278754e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.92745308278754e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.92745308278754e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911598693e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911598693e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.28766062480059e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.28766062480059e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.28766062480059e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.28766062480059e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.28766062480059e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.28766062480059e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.28766062480059e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.28766062480059e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751076324e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751076324e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329304274e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329304274e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350547581e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350547581e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265654496033e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265654496033e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265654496033e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265654496033e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128892066e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128892066e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480446914e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480446914e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480446914e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480446914e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516183731392e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516183731392e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412772575496e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412772575496e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412772575496e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412772575496e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915555489e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915555489e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915555489e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915555489e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176266304e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176266304e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176266304e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176266304e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148142585e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148142585e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148142585e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148142585e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148142585e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148142585e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148142585e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148142585e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781481425848e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781481425848e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781481425848e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781481425848e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694863976733e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694863976733e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599635103e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599635103e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599635103e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599635103e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599635103e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599635103e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599635103e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599635103e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594607263e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594607263e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594607263e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594607263e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310136022134e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310136022134e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310136022134e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310136022134e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915555489e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915555489e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915555489e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915555489e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516183731392e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516183731392e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128892066e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128892066e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961646905e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961646905e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961646905e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961646905e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350547581e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350547581e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329304274e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329304274e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751076324e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751076324e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911598693e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911598693e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201664051e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201664051e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.9568953730526e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.9568953730526e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651947274e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651947274e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651947274e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651947274e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601368e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601368e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824928426e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824928426e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217396877e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217396877e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217396877e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217396877e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887005826e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887005826e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490841362e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490841362e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015721818e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015721818e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507113275385e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507113275385e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946300534e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946300534e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.15134631114171e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.15134631114171e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500156619e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500156619e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312893737307e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312893737307e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294545135e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294545135e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.18393255939718e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.18393255939718e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692181757895e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692181757895e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068509867e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068509867e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122202625e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122202625e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713469582e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713469582e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110201) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110201) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110201) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110201) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916396) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916396) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499818) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499818) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499818) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499818) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125476) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125476) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213706) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213706) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213706) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213706) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440516) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440516) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440516) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440516) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369555) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369555) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630338) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630338) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524654) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524654) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339183) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339183) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339183) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339183) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.0039615607924965174) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.0039615607924965174) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.0039615607924965174) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.0039615607924965174) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.0044248554494418675) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.0044248554494418675) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639202) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639202) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.0046686203187763006) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.0046686203187763006) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155223) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155223) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221714) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221714) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221714) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221714) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109604) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109604) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109604) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109604) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592159) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592159) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592159) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592159) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381043) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381043) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269463) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269463) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269463) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269463) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158481) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158481) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158481) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158481) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671508) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671508) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671508) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671508) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.01096007494054266) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.01096007494054266) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.01096007494054266) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.01096007494054266) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848124) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848124) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130934) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130934) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130934) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130934) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226563) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226563) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226563) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226563) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238019) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238019) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238019) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238019) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375612) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375612) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375612) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375612) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303998) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303998) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303998) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303998) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535568) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535568) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535568) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535568) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535568) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535568) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535568) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535568) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068914) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068914) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068914) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068914) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068914) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068914) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068914) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068914) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149582) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149582) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149582) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149582) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884455) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884455) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884455) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884455) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914398) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914398) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129786) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129786) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780778) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780778) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780778) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780778) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613664) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613664) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613664) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613664) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928488345e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928488345e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277928488343e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928488343e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.595086006994783e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086006994783e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860069947818e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860069947818e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378232) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378232) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378232) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378232) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638305) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638305) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638305) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638305) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821706) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821706) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821706) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821706) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289329) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289329) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289329) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289329) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022052936) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022052936) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022052936) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022052936) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719752) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719752) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719752) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719752) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831251) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831251) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624776) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624776) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624776) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624776) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190546) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190546) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190546) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190546) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602679) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602679) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602679) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602679) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890873) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890873) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890873) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890873) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469287) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469287) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929528926) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929528926) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601307) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601307) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600825) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600825) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600825) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600825) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019028242443847203) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847203) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942878) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942878) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942878) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942878) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917959) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917959) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226563) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226563) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162063) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162063) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173048) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173048) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819217) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819217) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840945) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840945) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962584) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962584) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847198) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847198) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847198) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847198) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023949) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023949) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832954) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832954) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00592379833656134) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.00592379833656134) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017333) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017333) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109604) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109604) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840032) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840032) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832879) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832879) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832879) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832879) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235463) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235463) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235463) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235463) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025542) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025542) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806598) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806598) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806598) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806598) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352466) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352466) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352466) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352466) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696429) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696429) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696429) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696429) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696429) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696429) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696429) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696429) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569576194) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569576194) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549954) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549954) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549954) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549954) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880588069e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880588069e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305908143e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305908143e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305908143e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305908143e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795470793e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795470793e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795470793e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795470793e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775237077e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775237077e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775237077e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775237077e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.08979946766198e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.08979946766198e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.08979946766198e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.08979946766198e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669489086e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669489086e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669489086e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669489086e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833933864e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833933864e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833933864e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833933864e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736440782e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736440782e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736440782e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736440782e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038796296e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038796296e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038796296e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038796296e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147312878e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147312878e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147312878e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147312878e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.25322422557288e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.25322422557288e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594520054206e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594520054206e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954293383966e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954293383966e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954293383966e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954293383966e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954293383966e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954293383966e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954293383966e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954293383966e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320349101e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320349101e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320349101e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320349101e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604680292e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604680292e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604680292e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604680292e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098158827e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098158827e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098158827e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098158827e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836747666e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836747666e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836747666e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836747666e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771753326e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174771753326e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771753326e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174771753326e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675932292e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675932292e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675932292e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675932292e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675932292e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675932292e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675932292e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675932292e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824928426e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824928426e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824928426e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824928426e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288895443e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288895443e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288895443e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288895443e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104373478e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104373478e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104373478e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104373478e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975310733e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975310733e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206975857e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206975857e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744697864e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744697864e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471801085835e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471801085835e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471801085835e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471801085835e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677761927e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677761927e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231087868616e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231087868616e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231087868616e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231087868616e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350547581e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350547581e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350547581e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350547581e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265654496033e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265654496033e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595723336e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595723336e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595723336e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595723336e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480446914e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480446914e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915555489e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915555489e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594607263e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594607263e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096225647e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096225647e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096225647e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096225647e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594607263e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594607263e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.20935065214647e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.20935065214647e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.20935065214647e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.20935065214647e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555522267e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555522267e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555522267e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555522267e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915555489e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915555489e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480446914e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480446914e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265654496033e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265654496033e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677761927e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677761927e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744697864e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744697864e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206975857e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206975857e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975310733e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975310733e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887005826e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887005826e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887005826e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887005826e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435484806e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435484806e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435484806e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435484806e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514974073e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514974073e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514974073e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514974073e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184004488526e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184004488526e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184004488526e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184004488526e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184004488526e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184004488526e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184004488526e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184004488526e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420190906363e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190906363e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190906363e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190906363e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190906363e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190906363e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420190906363e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190906363e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500156619e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500156619e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500156619e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500156619e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893737307e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893737307e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.18393255939718e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.18393255939718e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880588069e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880588069e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569576194) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569576194) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409247) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409247) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409247) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409247) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005403) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005403) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005403) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005403) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005403) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005403) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005403) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005403) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125475) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125475) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125475) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125475) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907536) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907536) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907536) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907536) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349666) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349666) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349666) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349666) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127008) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127008) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127008) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127008) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823438) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823438) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823438) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823438) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823438) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823438) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823438) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823438) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619299) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619299) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619299) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619299) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840032) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840032) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0043110385079143) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.0043110385079143) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.0043110385079143) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.0043110385079143) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182544) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182544) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182544) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182544) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660393) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660393) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660393) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660393) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660393) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660393) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660393) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660393) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803851) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803851) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803851) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803851) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076841) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076841) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076841) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076841) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109604) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109604) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0053799371558393505) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.0053799371558393505) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.0053799371558393505) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.0053799371558393505) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017333) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017333) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960935) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960935) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960935) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960935) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.00592379833656134) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.00592379833656134) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832954) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832954) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023949) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023949) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962584) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962584) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840945) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840945) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819217) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819217) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173048) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173048) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162063) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162063) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226563) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226563) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917959) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917959) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847203) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847203) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.04587947078129786) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129786) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615604) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615604) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615603) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615603) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022716) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022716) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022716) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022716) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036474) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036474) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036474) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036474) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986362) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986362) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986362) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986362) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0763502195063499) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0763502195063499) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0763502195063499) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0763502195063499) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214009) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214009) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214009) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214009) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831251) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831251) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366184) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366184) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366184) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366184) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088383) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088383) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692875) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692875) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952893) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952893) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601307) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601307) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314618) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314618) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314618) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314618) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898734) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898734) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898734) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898734) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917959) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917959) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917959) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917959) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831839) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831839) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831839) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831839) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962584) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962584) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962584) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962584) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209829) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209829) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209829) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209829) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454816) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454816) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023949) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023949) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023949) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023949) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.0046686203187763006) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0046686203187763006) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336937) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336937) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728545) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728545) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728545) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728545) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217891) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217891) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832879) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832879) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235463) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235463) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101625) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101625) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369555) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369555) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312402) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312402) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169414) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169414) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169414) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169414) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024455) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024455) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487713) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487713) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756592) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756592) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549954) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549954) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221159171e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221159171e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221159171e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221159171e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736440781e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736440781e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.15134631114171e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.15134631114171e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507113275385e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507113275385e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063501848e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063501848e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713871816e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713871816e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320349101e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320349101e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562236084e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562236084e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507502825e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507502825e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507502825e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507502825e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103190205e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103190205e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103190205e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103190205e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.09163719898756e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09163719898756e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09163719898756e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.09163719898756e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.09163719898756e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09163719898756e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09163719898756e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.09163719898756e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986000125e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986000125e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986000125e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986000125e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986277192e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986277192e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986277192e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986277192e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104373477e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104373477e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464905559e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464905559e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464905559e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464905559e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464905559e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464905559e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464905559e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464905559e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421995164e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421995164e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421995164e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421995164e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421995164e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421995164e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421995164e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421995164e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521225633e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521225633e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521225633e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521225633e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085152656e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085152656e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085152656e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085152656e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085152656e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085152656e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085152656e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085152656e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595723336e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595723336e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381545472186e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381545472186e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555522267e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555522267e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.20935065214647e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.20935065214647e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244870206e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244870206e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244870206e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244870206e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244870206e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244870206e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244870206e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244870206e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.97422537959484e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.97422537959484e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.97422537959484e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.97422537959484e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716556035382e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716556035382e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716556035382e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716556035382e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.20935065214647e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.20935065214647e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183916175e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183916175e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183916175e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183916175e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494011065e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494011065e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494011065e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494011065e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555522267e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555522267e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052669828e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052669828e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052669828e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052669828e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381545472186e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381545472186e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595723336e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595723336e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506161525244e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161525244e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161525244e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161525244e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161525244e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161525244e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506161525244e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161525244e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978542256564e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978542256564e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978542256564e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978542256564e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095304526e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095304526e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095304526e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095304526e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425367476e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425367476e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425367476e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425367476e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425367476e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425367476e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425367476e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425367476e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104373477e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104373477e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562236084e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562236084e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320349101e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320349101e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713871816e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713871816e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765761096456e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765761096456e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011732396e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011732396e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011732396e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011732396e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063501848e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063501848e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507113275385e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507113275385e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.15134631114171e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.15134631114171e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016713327285e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016713327285e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016713327285e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016713327285e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736440781e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736440781e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722113135e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722113135e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722113135e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722113135e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327556337e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327556337e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327556337e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327556337e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502026196e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502026196e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502026196e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502026196e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656573415e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656573415e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656573415e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656573415e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718082581e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718082581e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718082581e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718082581e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.25327334824905e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.25327334824905e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793500315e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793500315e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793500315e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793500315e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218181e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411218181e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411218181e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218181e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549954) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549954) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389546053) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389546053) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389546053) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389546053) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756592) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756592) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569576194) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569576194) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569576194) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569576194) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487713) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487713) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908537) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908537) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908537) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908537) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024455) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024455) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230729945) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230729945) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230729945) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230729945) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312402) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312402) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369555) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369555) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415885) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415885) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415885) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415885) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235463) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235463) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832879) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832879) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217891) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217891) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336937) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336937) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.0046686203187763006) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0046686203187763006) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882780576) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882780576) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882780576) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882780576) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226829) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226829) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226829) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226829) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224099305) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224099305) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224099305) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224099305) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.00592379833656134) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379833656134) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.00592379833656134) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379833656134) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796759) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796759) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796759) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796759) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908943) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908943) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908943) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908943) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162063) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162063) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162063) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162063) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936376) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936376) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936376) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936376) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936376) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936376) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936376) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936376) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861636) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861636) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505273079826e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505273079826e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505273079846e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505273079846e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0716503518100253) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0716503518100253) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100253) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100253) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.010311482489831839) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831839) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209829) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209829) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770613) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770613) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770613) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770613) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311882) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311882) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311882) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311882) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311882) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311882) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311882) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311882) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676622) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676622) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676622) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676622) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285446) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285446) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121929) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121929) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121929) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121929) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415885) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415885) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093991) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093991) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093991) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093991) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101625) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101625) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587312) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587312) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587312) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587312) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587312) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587312) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587312) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587312) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312402) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312402) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312402) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312402) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538286) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538286) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538286) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538286) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538286) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538286) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538286) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538286) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562628) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562628) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562628) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562628) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146306145326713e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.146306145326713e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713871816e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713871816e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713871816e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713871816e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562236084e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562236084e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562236084e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562236084e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298316471e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298316471e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298316471e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298316471e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230256574e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230256574e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230256574e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230256574e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037396256e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037396256e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037396256e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037396256e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213315086e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213315086e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213315086e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213315086e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413670454e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413670454e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975310733e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975310733e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658351521e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658351521e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658351521e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658351521e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206975857e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206975857e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677761927e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677761927e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325323159776e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325323159776e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325323159776e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325323159776e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.01347145886439e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.01347145886439e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998846460184e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998846460184e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998846460184e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998846460184e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731755022121e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731755022121e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731755022121e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731755022121e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928603167e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928603167e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316092087e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309316092087e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316092087e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309316092087e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928603167e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928603167e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381545472186e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381545472186e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381545472186e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381545472186e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.01347145886439e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.01347145886439e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677761927e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677761927e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023904735983e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023904735983e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023904735983e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023904735983e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206975857e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206975857e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975310733e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975310733e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413670454e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413670454e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487116929e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487116929e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577359067e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577359067e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577359067e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577359067e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576109645e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576109645e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063501848e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063501848e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063501848e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063501848e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.25327334824905e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.25327334824905e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.401710973546222e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.401710973546222e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.401710973546222e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.401710973546222e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693198127e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693198127e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693198127e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693198127e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487713) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487713) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487713) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487713) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024457) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024457) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024457) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024457) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441907) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441907) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441907) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441907) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019244925) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019244925) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019244925) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019244925) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004533) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004533) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004533) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004533) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980192) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980192) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980192) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980192) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980192) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980192) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980192) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980192) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415885) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415885) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285446) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285446) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336937) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336937) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336937) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336937) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046421) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046421) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046421) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046421) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209829) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209829) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831839) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831839) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861636) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861636) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009018389947e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009018389947e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009018389947e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009018389947e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217891) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217891) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121929) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121929) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756592) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756592) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.146306145326713e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.146306145326713e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577359067e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577359067e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413670454e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413670454e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413670454e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413670454e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928603167e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928603167e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928603167e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928603167e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.01347145886439e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.01347145886439e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.01347145886439e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.01347145886439e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487116929e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487116929e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577359067e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577359067e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756592) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756592) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121929) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121929) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217891) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217891) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
/home/runner/work/qml/qml/demonstrations/tutorial_measurement_optimize.py:401: UserWarning: The init module will be deprecated soon, since templates can now provide a method that returns the shape of parameter tensors.
Expectation value of XYI =  0.022659767960222316
Expectation value of XIZ =  0.07715357869738937
[0.27361669 0.00898685 0.26297431 0.00732554 0.21720814 0.00116213
 0.22790267 0.00082366]
Expectation value of XYI =  0.02265976796022237
Expectation value of XIZ =  0.07715357869738931
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
[0.02265977 0.07715358]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[1])]
[PauliZ(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[0]) @ PauliZ(wires=[2])]
pennylane.qnodes.base.QuantumFunctionError: Only observables that are qubit-wise commuting
Pauli words can be returned on the same wire
Minimum number of QWC groupings found: 2
Group 0:
Y0 X2 X3
Y0 Y1 X2 X3
X2 X3
Group 1:
Z0 Z1 Z2
Z0 Z1 Z2 Z3
Z0
Z0 Z1
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
/home/runner/work/qml/qml/demonstrations/tutorial_measurement_optimize.py:738: UserWarning: The init module will be deprecated soon, since templates can now provide a method that returns the shape of parameter tensors.
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/grouping/transformations.py:178: UserWarning: The template decorator is deprecated and will be removed in release v0.20.0
Term expectation values:
 </code>
 </pre>
 </details>

---

## 33. tutorial_qgrnn.html <a name="demo32"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
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
Non-Existing Edge Parameters: [0.03795849267891518, -0.0022991817976616685]
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

## 34. tutorial_quanvolution.html <a name="demo33"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 3473408/11490434 [========>.....................] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step
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

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 4112384/11490434 [=========>....................] - ETA: 0s
 4202496/11490434 [=========>....................] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step
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
 </code>
 </pre>
 </details>

---

## 35. tutorial_rotoselect.html <a name="demo34"></a>

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

## 36. tutorial_quantum_natural_gradient.html <a name="demo35"></a>

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

