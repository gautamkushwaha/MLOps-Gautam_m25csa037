#MLOps_Gautam_m25csa037
# 1(a)

# MINST Dataset
<table>
  <thead>
    <tr>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Optimizers</th>
      <th rowspan="2">Learning Rate</th>
      <th colspan="2">Test Classification Accuracy (in %)</th>
    </tr>
    <tr>
      <th>ResNet-18</th>
      <th>ResNet-50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>16</b></td>
      <td><b>SGD</b></td>
      <td>0.001</td>
      <td>98.82</td>
      <td>98.70</td>
    </tr>
    <tr>
      <td><b>16</b></td>
      <td><b>SGD</b></td>
      <td>0.0001</td>
      <td>98.56</td>
      <td>97.61</td>
    </tr>
    <tr>
      <td><b>16</b></td>
      <td><b>Adam</b></td>
      <td>0.001</td>
      <td>99.08</td>
      <td>97.86</td>
    </tr>
    <tr>
      <td><b>16</b></td>
      <td><b>Adam</b></td>
      <td>0.0001</td>
      <td>99.09</td>
      <td>98.35</td>
    </tr>
    <tr>
      <td><b>32</b></td>
      <td><b>SGD</b></td>
      <td>0.001</td>
      <td>99.15</td>
      <td>98.95</td>
    </tr>
    <tr>
      <td><b>32 <br><span style="color:red">[Optional]</span></b></td>
      <td><b>SGD</b></td>
      <td>0.0001</td>
      <td>98.85</td>
      <td>98.10</td>
    </tr>
    <tr>
      <td><b>32 <br><span style="color:red">[Optional]</span></b></td>
      <td><b>Adam</b></td>
      <td>0.001</td>
      <td>99.25</td>
      <td>98.40</td>
    </tr>
    <tr>
      <td><b>32</b></td>
      <td><b>Adam</b></td>
      <td>0.0001</td>
      <td>99.30</td>
      <td>98.75</td>
    </tr>
  </tbody>
</table>


# FashionMinst Dataset

<table>
  <thead>
    <tr>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Optimizers</th>
      <th rowspan="2">Learning Rate</th>
      <th colspan="2">Test Classification Accuracy (in %)</th>
    </tr>
    <tr>
      <th>ResNet-18</th>
      <th>ResNet-50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>16</b></td>
      <td><b>SGD</b></td>
      <td>0.001</td>
      <td>88.48</td>
      <td>85.24</td>
    </tr>
    <tr>
      <td><b>16</b></td>
      <td><b>SGD</b></td>
      <td>0.0001</td>
      <td>87.91</td>
      <td>80.52</td>
    </tr>
    <tr>
      <td><b>16</b></td>
      <td><b>Adam</b></td>
      <td>0.001</td>
      <td>88.05</td>
      <td>85.69</td>
    </tr>
    <tr>
      <td><b>16</b></td>
      <td><b>Adam</b></td>
      <td>0.0001</td>
      <td>88.46</td>
      <td>84.70</td>
    </tr>
    <tr>
      <td><b>32</b></td>
      <td><b>SGD</b></td>
      <td>0.001</td>
      <td>88.84</td>
      <td>85.34</td>
    </tr>
    <tr>
      <td><b>32 <br><span style="color:red">[Optional]</span></b></td>
      <td><b>SGD</b></td>
      <td>0.0001</td>
      <td>86.02</td>
      <td>79.87</td>
    </tr>
    <tr>
      <td><b>32 <br><span style="color:red">[Optional]</span></b></td>
      <td><b>Adam</b></td>
      <td>0.001</td>
      <td>88.68</td>
      <td>87.58</td>
    </tr>
    <tr>
      <td><b>32</b></td>
      <td><b>Adam</b></td>
      <td>0.0001</td>
      <td>88.50</td>
      <td>83.81</td>
    </tr>
  </tbody>
</table>


# 1(b)

 <p>Best Performing Configurations:
For MNIST:

Best Accuracy: 95.00% (Poly kernel, C=0.1 or 1.0, gamma=0.1, degree=2)

Training Time: ~9,300-9,500 ms

Test Time: ~2,100-2,300 ms

For FashionMNIST:

Best Accuracy: 87.75% (Poly kernel, C=0.1, gamma=0.01, degree=2)

Training Time: 7,130 ms

Test Time: 2,416 ms</p>

<h3>SVM Classifier Performance Summary</h3>
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Best Kernel</th>
      <th>Best Hyperparameters</th>
      <th>Test Accuracy (%)</th>
      <th>Training Time (ms)</th>
      <th>Testing Time (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>MNIST</b></td>
      <td>Polynomial</td>
      <td>C=0.1/1.0, gamma=0.1, degree=2</td>
      <td><b>95.00</b></td>
      <td>9,304 - 9,538</td>
      <td>2,116 - 2,266</td>
    </tr>
    <tr>
      <td><b>FashionMNIST</b></td>
      <td>Polynomial</td>
      <td>C=0.1, gamma=0.01, degree=2</td>
      <td><b>87.75</b></td>
      <td>7,130</td>
      <td>2,416</td>
    </tr>
  </tbody>
</table>

<h3>Comparison with Deep Learning Models:</h3>
<h3>SVM vs Deep Learning Performance Comparison</h3>
<table border="1">
  <thead>
    <tr>
      <th>Model</th>
      <th>MNIST Accuracy (%)</th>
      <th>FashionMNIST Accuracy (%)</th>
      <th>Training Time</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>SVM (Best)</b></td>
      <td>95.00</td>
      <td>87.75</td>
      <td>5-15 seconds</td>
      <td>Fast training, lower accuracy</td>
    </tr>
    <tr>
      <td><b>ResNet-18 (Best)</b></td>
      <td>99.30</td>
      <td>88.84</td>
      <td>20-45 seconds/epoch</td>
      <td>Higher accuracy, longer training</td>
    </tr>
    <tr>
      <td><b>ResNet-50 (Best)</b></td>
      <td>98.75</td>
      <td>87.58</td>
      <td>40-90 seconds/epoch</td>
      <td>Similar accuracy to SVM, much slower</td>
    </tr>
  </tbody>
</table>