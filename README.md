#MLOps_Gautam_m25csa037

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