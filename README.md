#MLOps_Gautam_m25csa037


<style>
table {
  border-collapse: collapse;
  width: 100%;
  margin: 20px 0;
  font-family: Arial, sans-serif;
}

th, td {
  border: 1px solid #ddd;
  padding: 12px;
  text-align: center;
}

th {
  background-color: #4CAF50;
  color: white;
  font-weight: bold;
}

tr:nth-child(even) {
  background-color: #f2f2f2;
}

tr:hover {
  background-color: #ddd;
}

b {
  color: #333;
}

span[style="color:red"] {
  font-size: 0.8em;
}
</style>

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