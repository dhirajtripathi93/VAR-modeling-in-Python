# VAR-modeling-in-Python
We analyze multiple affecting factors like time and rankings to forecast sales using Vector Auto-regression model in Python


## Jupyter Notebook

### AppFigures data challenge
#### Submitted by : Dhiraj Tripathi , Master's of IT & Analytics, Rutgers University - NJ
#### email: dhiraj.tripathi@rutgers.edu , contact : - 551-358-9117

### Problem Statement: 

In this data challenge, we are given two different datasets as 'ranks.csv' and 'sales.csv' . Our primary goal is to analyze the tend between the ranks and sales. The thought process here is to estimate sales of an application by considering ranks as a function. We will feed in some data to our machine learning model and forecast the sales value 3 months ahead.

To achieve this, we will start by importing some required packages for our script as below:


```python
import pandas as pd
```

### Reading the Csv:

Let's read the two csv files given to us and have a quick look at the dimensions and data types of the columns from these files.


```python
import numpy as np

rankdata = pd.read_csv("F:/GIT/Appfigues/ranks.csv" , index_col = False)
salesdata = pd.read_csv("F:/GIT/Appfigues/sales.csv", index_col = False)
```


```python
rankdata.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>app_id</th>
      <th>r0</th>
      <th>r1</th>
      <th>r2</th>
      <th>r3</th>
      <th>r4</th>
      <th>r5</th>
      <th>r6</th>
      <th>r7</th>
      <th>...</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>cs1</th>
      <th>cs2</th>
      <th>cs3</th>
      <th>cs4</th>
      <th>cs5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>1</td>
      <td>622.0</td>
      <td>622.0</td>
      <td>621.0</td>
      <td>619.0</td>
      <td>621.0</td>
      <td>561.0</td>
      <td>562.0</td>
      <td>563.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-01</td>
      <td>2</td>
      <td>574.0</td>
      <td>574.0</td>
      <td>573.0</td>
      <td>571.0</td>
      <td>573.0</td>
      <td>543.0</td>
      <td>544.0</td>
      <td>545.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>142.0</td>
      <td>88.0</td>
      <td>296.0</td>
      <td>826.0</td>
      <td>3177.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-01</td>
      <td>4</td>
      <td>144.0</td>
      <td>144.0</td>
      <td>144.0</td>
      <td>144.0</td>
      <td>144.0</td>
      <td>144.0</td>
      <td>143.0</td>
      <td>142.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>552.0</td>
      <td>321.0</td>
      <td>705.0</td>
      <td>1409.0</td>
      <td>4417.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-01</td>
      <td>5</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>918.0</td>
      <td>914.0</td>
      <td>917.0</td>
      <td>875.0</td>
      <td>874.0</td>
      <td>877.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>226.0</td>
      <td>95.0</td>
      <td>322.0</td>
      <td>1121.0</td>
      <td>4013.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-01</td>
      <td>8</td>
      <td>903.0</td>
      <td>903.0</td>
      <td>902.0</td>
      <td>898.0</td>
      <td>901.0</td>
      <td>920.0</td>
      <td>919.0</td>
      <td>922.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>356.0</td>
      <td>181.0</td>
      <td>362.0</td>
      <td>1744.0</td>
      <td>6884.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
salesdata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>app_id</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>320</td>
      <td>2412</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-01</td>
      <td>406</td>
      <td>1308</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-01</td>
      <td>459</td>
      <td>2037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-01</td>
      <td>722</td>
      <td>2052</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-01</td>
      <td>1234</td>
      <td>1553</td>
    </tr>
  </tbody>
</table>
</div>




```python
rankdata.ndim

```




    2




```python
rankdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 97608 entries, 0 to 97607
    Data columns (total 36 columns):
    Date      97608 non-null object
    app_id    97608 non-null int64
    r0        89992 non-null float64
    r1        89953 non-null float64
    r2        88974 non-null float64
    r3        89999 non-null float64
    r4        89997 non-null float64
    r5        89886 non-null float64
    r6        89888 non-null float64
    r7        89991 non-null float64
    r8        89994 non-null float64
    r9        89995 non-null float64
    r10       89977 non-null float64
    r11       89997 non-null float64
    r12       89958 non-null float64
    r13       89931 non-null float64
    r14       89990 non-null float64
    r15       89998 non-null float64
    r16       89960 non-null float64
    r17       89984 non-null float64
    r18       89991 non-null float64
    r19       89973 non-null float64
    r20       89993 non-null float64
    r21       89999 non-null float64
    r22       89995 non-null float64
    r23       89981 non-null float64
    s1        93967 non-null float64
    s2        93967 non-null float64
    s3        93967 non-null float64
    s4        93967 non-null float64
    s5        93967 non-null float64
    cs1       93967 non-null float64
    cs2       93967 non-null float64
    cs3       93967 non-null float64
    cs4       93967 non-null float64
    cs5       93967 non-null float64
    dtypes: float64(34), int64(1), object(1)
    memory usage: 26.8+ MB
    


```python
rankdata.shape
```




    (97608, 36)




```python
salesdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1105 entries, 0 to 1104
    Data columns (total 3 columns):
    Date      1105 non-null object
    app_id    1105 non-null int64
    sales     1105 non-null int64
    dtypes: int64(2), object(1)
    memory usage: 26.0+ KB
    


```python
salesdata.shape
```




    (1105, 3)



We can notice that our data sets are sufficiently large in terms of number of rows * columns and also there are some matching columns in both the datasets such as Date , app_id .

### Data Munging:

The first step towards creating a model is to merge both the datasets and extract only the relevant information from both. For this, we will use the merge function in python as if we are using an inner join in sql.


```python
subsetDataFrame = pd.merge(rankdata, salesdata, how='inner', on=['Date', 'app_id'])

```


```python
print(subsetDataFrame)
```

                Date  app_id     r0     r1     r2     r3     r4     r5     r6  \
    0     2016-01-01     320  488.0  488.0  488.0  486.0  487.0  475.0  476.0   
    1     2016-01-01     406  685.0  685.0  684.0  681.0  684.0  702.0  701.0   
    2     2016-01-01     459  597.0  597.0  596.0  594.0  596.0  603.0  604.0   
    3     2016-01-01     722  532.0  532.0  532.0  530.0  531.0  547.0  548.0   
    4     2016-01-01    1234  813.0  813.0  812.0  809.0  812.0  831.0  830.0   
    5     2016-01-01    1490  528.0  528.0  528.0  526.0  527.0  519.0  520.0   
    6     2016-01-01    2398    NaN    NaN    NaN    NaN    NaN    NaN    NaN   
    7     2016-01-01    2891    5.0    5.0    5.0    5.0    5.0    5.0    6.0   
    8     2016-01-02     320  438.0  435.0  446.0  445.0  446.0  444.0  446.0   
    9     2016-01-02     406  738.0  735.0  747.0  748.0  749.0  746.0  749.0   
    10    2016-01-02     459  610.0  607.0  582.0  581.0  582.0  580.0  582.0   
    11    2016-01-02     722  601.0  598.0  603.0  603.0  604.0  602.0  604.0   
    12    2016-01-02    1234  678.0  675.0  698.0  698.0  699.0  696.0  699.0   
    13    2016-01-02    1490  535.0  532.0  564.0  563.0  564.0  562.0  564.0   
    14    2016-01-02    2346    NaN    NaN    NaN    NaN    NaN    NaN    NaN   
    15    2016-01-02    2398  950.0  947.0  969.0  970.0  972.0  967.0  972.0   
    16    2016-01-02    2891    6.0    6.0    6.0    6.0    6.0    6.0    6.0   
    17    2016-01-03     320  492.0  492.0  492.0  491.0  491.0  492.0  504.0   
    18    2016-01-03     406  740.0  737.0  740.0  738.0  739.0  740.0  728.0   
    19    2016-01-03     459  558.0  558.0  558.0  557.0  557.0  558.0  568.0   
    20    2016-01-03     722  647.0  646.0  647.0  645.0  646.0  647.0  679.0   
    21    2016-01-03    1234  809.0  806.0  809.0  807.0  808.0  809.0  836.0   
    22    2016-01-03    1490  533.0  533.0  533.0  532.0  532.0  533.0  564.0   
    23    2016-01-03    2346  821.0  818.0  821.0  819.0  820.0  821.0  702.0   
    24    2016-01-03    2891    5.0    5.0    5.0    5.0    5.0    5.0    5.0   
    25    2016-01-04     320  521.0  521.0  521.0  521.0  521.0  521.0  521.0   
    26    2016-01-04     406  816.0  816.0  816.0  816.0  816.0  816.0  816.0   
    27    2016-01-04     459  576.0  576.0  576.0  576.0  576.0  576.0  576.0   
    28    2016-01-04     722  707.0  707.0  707.0  707.0  707.0  707.0  707.0   
    29    2016-01-04    1234  844.0  844.0  844.0  844.0  844.0  844.0  844.0   
    ...          ...     ...    ...    ...    ...    ...    ...    ...    ...   
    1075  2016-03-30    1264    NaN    NaN    NaN    NaN    NaN    NaN    NaN   
    1076  2016-03-30    1461  960.0  967.0  967.0  967.0  967.0  967.0  967.0   
    1077  2016-03-30    1490  317.0  324.0  324.0  324.0  324.0  324.0  324.0   
    1078  2016-03-30    1874  584.0  591.0  591.0  591.0  591.0  591.0  591.0   
    1079  2016-03-30    2346  307.0  314.0  314.0  314.0  314.0  314.0  314.0   
    1080  2016-03-30    2373  969.0  976.0  976.0  976.0  976.0  976.0  976.0   
    1081  2016-03-30    2398  856.0  863.0  863.0  863.0  863.0  863.0  863.0   
    1082  2016-03-30    2667  460.0  467.0  467.0  467.0  467.0  467.0  467.0   
    1083  2016-03-30    2891   17.0   18.0   18.0   18.0   18.0   18.0   18.0   
    1084  2016-03-30    3308    NaN    NaN    NaN    NaN    NaN    NaN    NaN   
    1085  2016-03-30    3373  252.0  259.0  259.0  259.0  259.0  259.0  259.0   
    1086  2016-03-30    3428  568.0  575.0  575.0  575.0  575.0  575.0  575.0   
    1087  2016-03-30    3550  537.0  544.0  544.0  544.0  544.0  544.0  544.0   
    1088  2016-03-31     320  444.0  445.0  445.0  445.0  445.0  445.0  445.0   
    1089  2016-03-31     459  581.0  498.0  498.0  498.0  498.0  498.0  498.0   
    1090  2016-03-31     676  647.0  633.0  633.0  633.0  633.0  633.0  633.0   
    1091  2016-03-31     722  625.0  658.0  658.0  658.0  658.0  658.0  658.0   
    1092  2016-03-31    1234  517.0  512.0  512.0  512.0  512.0  512.0  512.0   
    1093  2016-03-31    1264  989.0  977.0  977.0  977.0  977.0  977.0  977.0   
    1094  2016-03-31    1461  882.0  884.0  884.0  884.0  884.0  884.0  884.0   
    1095  2016-03-31    1490  547.0  587.0  587.0  587.0  587.0  587.0  587.0   
    1096  2016-03-31    1874  545.0  571.0  571.0  571.0  571.0  571.0  571.0   
    1097  2016-03-31    2346  362.0  377.0  377.0  377.0  377.0  377.0  377.0   
    1098  2016-03-31    2398  935.0  916.0  916.0  916.0  916.0  916.0  916.0   
    1099  2016-03-31    2667  475.0  488.0  488.0  488.0  488.0  488.0  488.0   
    1100  2016-03-31    2891   20.0   21.0   21.0   21.0   21.0   21.0   21.0   
    1101  2016-03-31    3308  964.0  904.0  904.0  904.0  904.0  904.0  904.0   
    1102  2016-03-31    3373  282.0  310.0  310.0  310.0  310.0  310.0  310.0   
    1103  2016-03-31    3428  554.0  573.0  573.0  573.0  573.0  573.0  573.0   
    1104  2016-03-31    3550  274.0  217.0  217.0  217.0  217.0  217.0  217.0   
    
             r7  ...     s2    s3    s4     s5     cs1     cs2     cs3      cs4  \
    0     477.0  ...    0.0   2.0   0.0    0.0   746.0   284.0   290.0    568.0   
    1     704.0  ...    0.0   0.0   0.0    0.0  1376.0  1008.0  5674.0  20010.0   
    2     605.0  ...    0.0   0.0   0.0    1.0    10.0     2.0    14.0     32.0   
    3     549.0  ...    0.0   2.0   8.0    1.0  4025.0  1129.0  1488.0   2007.0   
    4     833.0  ...    0.0   0.0   0.0    0.0   332.0   163.0   269.0    395.0   
    5     521.0  ...    0.0   0.0   0.0    2.0   248.0    91.0   167.0    290.0   
    6       NaN  ...    0.0   0.0   0.0    0.0    40.0    13.0    17.0     24.0   
    7       6.0  ...    4.0   8.0  16.0  106.0    73.0    41.0    71.0    191.0   
    8     446.0  ...    0.0   0.0   0.0    0.0   746.0   284.0   290.0    568.0   
    9     749.0  ...    0.0   0.0   0.0    0.0  1376.0  1008.0  5674.0  20010.0   
    10    582.0  ...    0.0   0.0   0.0    1.0    11.0     2.0    14.0     32.0   
    11    604.0  ...    0.0   0.0   2.0    2.0  4027.0  1129.0  1488.0   2009.0   
    12    699.0  ...    0.0   0.0   0.0    0.0   332.0   163.0   269.0    395.0   
    13    564.0  ...    0.0   2.0   0.0    0.0   248.0    91.0   169.0    290.0   
    14      NaN  ...    0.0   0.0   0.0    1.0    21.0    30.0    51.0     74.0   
    15    972.0  ...    0.0   0.0   0.0    0.0    40.0    13.0    17.0     24.0   
    16      6.0  ...    0.0  16.0  14.0  111.0    79.0    41.0    87.0    205.0   
    17    503.0  ...    0.0   0.0   0.0    0.0   746.0   284.0   290.0    568.0   
    18    726.0  ...    0.0   0.0   0.0    0.0  1376.0  1008.0  5674.0  20010.0   
    19    566.0  ...    0.0   0.0   0.0    0.0    12.0     2.0    14.0     32.0   
    20    677.0  ...    2.0   0.0   0.0    0.0  4031.0  1131.0  1488.0   2009.0   
    21    834.0  ...    0.0   0.0   0.0    0.0   332.0   163.0   269.0    395.0   
    22    562.0  ...    0.0   0.0   0.0    0.0   250.0    91.0   169.0    290.0   
    23    700.0  ...    0.0   0.0   0.0    0.0    21.0    30.0    51.0     74.0   
    24      5.0  ...    4.0   6.0  26.0   61.0    85.0    45.0    93.0    231.0   
    25    530.0  ...    0.0   0.0   0.0    0.0   747.0   284.0   290.0    568.0   
    26    832.0  ...    0.0   0.0   0.0    0.0  1376.0  1008.0  5674.0  20010.0   
    27    568.0  ...    0.0   0.0   0.0    0.0    12.0     2.0    14.0     32.0   
    28    726.0  ...    0.0   0.0   0.0    2.0  4031.0  1131.0  1488.0   2009.0   
    29    856.0  ...    0.0   0.0   0.0    0.0   332.0   163.0   269.0    395.0   
    ...     ...  ...    ...   ...   ...    ...     ...     ...     ...      ...   
    1075    NaN  ...    0.0   0.0   0.0    0.0   994.0   433.0   462.0    339.0   
    1076  975.0  ...    0.0   0.0   0.0    2.0   102.0    85.0   243.0   1104.0   
    1077  386.0  ...    0.0   1.0   2.0    4.0   282.0   108.0   202.0    341.0   
    1078  523.0  ...    0.0   0.0   0.0    0.0    11.0     0.0     1.0      1.0   
    1079  335.0  ...    0.0   0.0   1.0    0.0    37.0    40.0    70.0    108.0   
    1080  939.0  ...    0.0   0.0   1.0    0.0    53.0    46.0    81.0    402.0   
    1081  896.0  ...    0.0   0.0   0.0    0.0    45.0    17.0    22.0     28.0   
    1082  446.0  ...    0.0   4.0   2.0    1.0   154.0   104.0   157.0    221.0   
    1083   18.0  ...    1.0   2.0   1.0   19.0   216.0   129.0   275.0    584.0   
    1084  998.0  ...    0.0   0.0   0.0    0.0     3.0     1.0     9.0      8.0   
    1085  272.0  ...    1.0   1.0   0.0    2.0     5.0     2.0     6.0     26.0   
    1086  643.0  ...    NaN   NaN   NaN    NaN     NaN     NaN     NaN      NaN   
    1087  537.0  ...    NaN   NaN   NaN    NaN     NaN     NaN     NaN      NaN   
    1088  445.0  ...    0.0   0.0   0.0    0.0   757.0   291.0   294.0    571.0   
    1089  509.0  ...    0.0   0.0   0.0    0.0    20.0     5.0    17.0     36.0   
    1090  619.0  ...    0.0   0.0   0.0    0.0   222.0   120.0   749.0   2727.0   
    1091  679.0  ...    0.0   0.0   0.0    2.0  4245.0  1175.0  1540.0   2121.0   
    1092  497.0  ...    0.0   0.0   0.0    0.0   344.0   169.0   270.0    404.0   
    1093  947.0  ...    0.0   1.0   1.0    1.0   995.0   433.0   463.0    340.0   
    1094  913.0  ...    0.0   0.0   0.0    2.0   102.0    85.0   243.0   1104.0   
    1095  639.0  ...    0.0   0.0   1.0    1.0   284.0   108.0   202.0    342.0   
    1096  583.0  ...    0.0   0.0   0.0    0.0    11.0     0.0     1.0      1.0   
    1097  394.0  ...    3.0   3.0   2.0    0.0    39.0    43.0    73.0    110.0   
    1098  919.0  ...    0.0   0.0   0.0    0.0    45.0    17.0    22.0     28.0   
    1099  476.0  ...    1.0   0.0   4.0    2.0   155.0   105.0   157.0    225.0   
    1100   22.0  ...    1.0   2.0   7.0   21.0   219.0   130.0   277.0    591.0   
    1101  959.0  ...    0.0   0.0   0.0    1.0     3.0     1.0     9.0      8.0   
    1102  314.0  ...    1.0   0.0   5.0    7.0     5.0     3.0     6.0     31.0   
    1103  656.0  ...    NaN   NaN   NaN    NaN     NaN     NaN     NaN      NaN   
    1104  195.0  ...    NaN   NaN   NaN    NaN     NaN     NaN     NaN      NaN   
    
              cs5  sales  
    0      1736.0   2412  
    1     82635.0   1308  
    2       264.0   2037  
    3      5477.0   2052  
    4      1320.0   1553  
    5       877.0   2152  
    6       120.0   1501  
    7      1036.0  89289  
    8      1736.0   1622  
    9     82635.0   1110  
    10      265.0   1553  
    11     5479.0   1467  
    12     1320.0   1011  
    13      877.0   1722  
    14      104.0   2165  
    15      120.0   1113  
    16     1147.0  84915  
    17     1736.0   1564  
    18    82635.0    963  
    19      265.0   1409  
    20     5479.0   1371  
    21     1320.0    938  
    22      877.0   1662  
    23      104.0   2029  
    24     1208.0  61345  
    25     1736.0   1609  
    26    82635.0   1168  
    27      265.0   1732  
    28     5481.0   1421  
    29     1320.0    989  
    ...       ...    ...  
    1075   1061.0   2654  
    1076   4328.0   1039  
    1077   1029.0   1117  
    1078     11.0   1364  
    1079    127.0   2926  
    1080   2462.0    757  
    1081    122.0   1297  
    1082    345.0   3257  
    1083   2411.0  36890  
    1084     48.0    856  
    1085     93.0   3859  
    1086      NaN   1135  
    1087      NaN   6737  
    1088   1742.0   2041  
    1089    323.0   1526  
    1090  12217.0   1157  
    1091   5742.0   1388  
    1092   1331.0   1800  
    1093   1062.0   2433  
    1094   4330.0   1022  
    1095   1030.0   1223  
    1096     11.0    237  
    1097    127.0  11099  
    1098    122.0   1372  
    1099    347.0   3294  
    1100   2432.0  35903  
    1101     49.0    777  
    1102    100.0   3770  
    1103      NaN   1095  
    1104      NaN   7912  
    
    [1105 rows x 37 columns]
    

### Data cleaning:

As we can see in the above dataframe , there are a lot of NaN values , which can distract our ML model. For the sake of simplicity , we will drop the records which have missing values in it. We can also impute the missing values, if required. But, our dataframe is large enough to train the model, even if we drop the records.


```python
subsetDataFrame.dropna(inplace=True)

```


```python
subsetDataFrame.shape
```




    (919, 37)




```python
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 20)  


```


```python
display(subsetDataFrame)


```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>app_id</th>
      <th>r0</th>
      <th>r1</th>
      <th>r2</th>
      <th>r3</th>
      <th>r4</th>
      <th>r5</th>
      <th>r6</th>
      <th>r7</th>
      <th>r8</th>
      <th>r9</th>
      <th>r10</th>
      <th>r11</th>
      <th>r12</th>
      <th>r13</th>
      <th>r14</th>
      <th>r15</th>
      <th>r16</th>
      <th>r17</th>
      <th>r18</th>
      <th>r19</th>
      <th>r20</th>
      <th>r21</th>
      <th>r22</th>
      <th>r23</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>cs1</th>
      <th>cs2</th>
      <th>cs3</th>
      <th>cs4</th>
      <th>cs5</th>
      <th>sales</th>
      <th>avgrank</th>
      <th>avgrating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>320</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>486.0</td>
      <td>487.0</td>
      <td>475.0</td>
      <td>476.0</td>
      <td>477.0</td>
      <td>478.0</td>
      <td>477.0</td>
      <td>478.0</td>
      <td>478.0</td>
      <td>478.0</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>472.0</td>
      <td>452.0</td>
      <td>452.0</td>
      <td>452.0</td>
      <td>452.0</td>
      <td>452.0</td>
      <td>452.0</td>
      <td>438.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>746.0</td>
      <td>284.0</td>
      <td>290.0</td>
      <td>568.0</td>
      <td>1736.0</td>
      <td>2412</td>
      <td>469.739130</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-01</td>
      <td>406</td>
      <td>685.0</td>
      <td>685.0</td>
      <td>684.0</td>
      <td>681.0</td>
      <td>684.0</td>
      <td>702.0</td>
      <td>701.0</td>
      <td>704.0</td>
      <td>705.0</td>
      <td>703.0</td>
      <td>717.0</td>
      <td>717.0</td>
      <td>717.0</td>
      <td>738.0</td>
      <td>738.0</td>
      <td>738.0</td>
      <td>738.0</td>
      <td>742.0</td>
      <td>742.0</td>
      <td>742.0</td>
      <td>742.0</td>
      <td>741.0</td>
      <td>742.0</td>
      <td>738.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1376.0</td>
      <td>1008.0</td>
      <td>5674.0</td>
      <td>20010.0</td>
      <td>82635.0</td>
      <td>1308</td>
      <td>719.173913</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-01</td>
      <td>459</td>
      <td>597.0</td>
      <td>597.0</td>
      <td>596.0</td>
      <td>594.0</td>
      <td>596.0</td>
      <td>603.0</td>
      <td>604.0</td>
      <td>605.0</td>
      <td>606.0</td>
      <td>604.0</td>
      <td>618.0</td>
      <td>618.0</td>
      <td>618.0</td>
      <td>607.0</td>
      <td>607.0</td>
      <td>607.0</td>
      <td>607.0</td>
      <td>598.0</td>
      <td>598.0</td>
      <td>598.0</td>
      <td>598.0</td>
      <td>598.0</td>
      <td>598.0</td>
      <td>610.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>264.0</td>
      <td>2037</td>
      <td>603.695652</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-01</td>
      <td>722</td>
      <td>532.0</td>
      <td>532.0</td>
      <td>532.0</td>
      <td>530.0</td>
      <td>531.0</td>
      <td>547.0</td>
      <td>548.0</td>
      <td>549.0</td>
      <td>550.0</td>
      <td>548.0</td>
      <td>574.0</td>
      <td>574.0</td>
      <td>574.0</td>
      <td>577.0</td>
      <td>577.0</td>
      <td>577.0</td>
      <td>577.0</td>
      <td>594.0</td>
      <td>594.0</td>
      <td>594.0</td>
      <td>594.0</td>
      <td>594.0</td>
      <td>594.0</td>
      <td>601.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>4025.0</td>
      <td>1129.0</td>
      <td>1488.0</td>
      <td>2007.0</td>
      <td>5477.0</td>
      <td>2052</td>
      <td>567.913043</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-01</td>
      <td>1234</td>
      <td>813.0</td>
      <td>813.0</td>
      <td>812.0</td>
      <td>809.0</td>
      <td>812.0</td>
      <td>831.0</td>
      <td>830.0</td>
      <td>833.0</td>
      <td>834.0</td>
      <td>832.0</td>
      <td>802.0</td>
      <td>802.0</td>
      <td>802.0</td>
      <td>775.0</td>
      <td>775.0</td>
      <td>775.0</td>
      <td>775.0</td>
      <td>713.0</td>
      <td>713.0</td>
      <td>713.0</td>
      <td>713.0</td>
      <td>712.0</td>
      <td>713.0</td>
      <td>678.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>332.0</td>
      <td>163.0</td>
      <td>269.0</td>
      <td>395.0</td>
      <td>1320.0</td>
      <td>1553</td>
      <td>776.826087</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-01-01</td>
      <td>1490</td>
      <td>528.0</td>
      <td>528.0</td>
      <td>528.0</td>
      <td>526.0</td>
      <td>527.0</td>
      <td>519.0</td>
      <td>520.0</td>
      <td>521.0</td>
      <td>522.0</td>
      <td>520.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>506.0</td>
      <td>522.0</td>
      <td>522.0</td>
      <td>522.0</td>
      <td>522.0</td>
      <td>522.0</td>
      <td>522.0</td>
      <td>535.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>248.0</td>
      <td>91.0</td>
      <td>167.0</td>
      <td>290.0</td>
      <td>877.0</td>
      <td>2152</td>
      <td>518.260870</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016-01-01</td>
      <td>2891</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>106.0</td>
      <td>73.0</td>
      <td>41.0</td>
      <td>71.0</td>
      <td>191.0</td>
      <td>1036.0</td>
      <td>89289</td>
      <td>5.782609</td>
      <td>33.50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016-01-02</td>
      <td>320</td>
      <td>438.0</td>
      <td>435.0</td>
      <td>446.0</td>
      <td>445.0</td>
      <td>446.0</td>
      <td>444.0</td>
      <td>446.0</td>
      <td>446.0</td>
      <td>453.0</td>
      <td>453.0</td>
      <td>452.0</td>
      <td>455.0</td>
      <td>455.0</td>
      <td>454.0</td>
      <td>470.0</td>
      <td>470.0</td>
      <td>484.0</td>
      <td>484.0</td>
      <td>484.0</td>
      <td>484.0</td>
      <td>492.0</td>
      <td>493.0</td>
      <td>490.0</td>
      <td>492.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>746.0</td>
      <td>284.0</td>
      <td>290.0</td>
      <td>568.0</td>
      <td>1736.0</td>
      <td>1622</td>
      <td>464.043478</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016-01-02</td>
      <td>406</td>
      <td>738.0</td>
      <td>735.0</td>
      <td>747.0</td>
      <td>748.0</td>
      <td>749.0</td>
      <td>746.0</td>
      <td>749.0</td>
      <td>749.0</td>
      <td>761.0</td>
      <td>761.0</td>
      <td>760.0</td>
      <td>766.0</td>
      <td>766.0</td>
      <td>765.0</td>
      <td>779.0</td>
      <td>780.0</td>
      <td>754.0</td>
      <td>756.0</td>
      <td>756.0</td>
      <td>756.0</td>
      <td>745.0</td>
      <td>746.0</td>
      <td>743.0</td>
      <td>740.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1376.0</td>
      <td>1008.0</td>
      <td>5674.0</td>
      <td>20010.0</td>
      <td>82635.0</td>
      <td>1110</td>
      <td>754.652174</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2016-01-02</td>
      <td>459</td>
      <td>610.0</td>
      <td>607.0</td>
      <td>582.0</td>
      <td>581.0</td>
      <td>582.0</td>
      <td>580.0</td>
      <td>582.0</td>
      <td>582.0</td>
      <td>584.0</td>
      <td>584.0</td>
      <td>583.0</td>
      <td>593.0</td>
      <td>593.0</td>
      <td>592.0</td>
      <td>587.0</td>
      <td>587.0</td>
      <td>604.0</td>
      <td>604.0</td>
      <td>604.0</td>
      <td>604.0</td>
      <td>571.0</td>
      <td>571.0</td>
      <td>568.0</td>
      <td>558.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>265.0</td>
      <td>1553</td>
      <td>586.217391</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>2016-03-31</td>
      <td>1264</td>
      <td>989.0</td>
      <td>977.0</td>
      <td>977.0</td>
      <td>977.0</td>
      <td>977.0</td>
      <td>977.0</td>
      <td>977.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>947.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>995.0</td>
      <td>433.0</td>
      <td>463.0</td>
      <td>340.0</td>
      <td>1062.0</td>
      <td>2433</td>
      <td>954.826087</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>2016-03-31</td>
      <td>1461</td>
      <td>882.0</td>
      <td>884.0</td>
      <td>884.0</td>
      <td>884.0</td>
      <td>884.0</td>
      <td>884.0</td>
      <td>884.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>913.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>102.0</td>
      <td>85.0</td>
      <td>243.0</td>
      <td>1104.0</td>
      <td>4330.0</td>
      <td>1022</td>
      <td>905.434783</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>2016-03-31</td>
      <td>1490</td>
      <td>547.0</td>
      <td>587.0</td>
      <td>587.0</td>
      <td>587.0</td>
      <td>587.0</td>
      <td>587.0</td>
      <td>587.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>639.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>284.0</td>
      <td>108.0</td>
      <td>202.0</td>
      <td>342.0</td>
      <td>1030.0</td>
      <td>1223</td>
      <td>625.434783</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>2016-03-31</td>
      <td>1874</td>
      <td>545.0</td>
      <td>571.0</td>
      <td>571.0</td>
      <td>571.0</td>
      <td>571.0</td>
      <td>571.0</td>
      <td>571.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>583.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>237</td>
      <td>579.869565</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>2016-03-31</td>
      <td>2346</td>
      <td>362.0</td>
      <td>377.0</td>
      <td>377.0</td>
      <td>377.0</td>
      <td>377.0</td>
      <td>377.0</td>
      <td>377.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>394.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>39.0</td>
      <td>43.0</td>
      <td>73.0</td>
      <td>110.0</td>
      <td>127.0</td>
      <td>11099</td>
      <td>389.565217</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>2016-03-31</td>
      <td>2398</td>
      <td>935.0</td>
      <td>916.0</td>
      <td>916.0</td>
      <td>916.0</td>
      <td>916.0</td>
      <td>916.0</td>
      <td>916.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>919.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>28.0</td>
      <td>122.0</td>
      <td>1372</td>
      <td>918.217391</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>2016-03-31</td>
      <td>2667</td>
      <td>475.0</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>488.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>476.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>155.0</td>
      <td>105.0</td>
      <td>157.0</td>
      <td>225.0</td>
      <td>347.0</td>
      <td>3294</td>
      <td>479.130435</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>2016-03-31</td>
      <td>2891</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>219.0</td>
      <td>130.0</td>
      <td>277.0</td>
      <td>591.0</td>
      <td>2432.0</td>
      <td>35903</td>
      <td>21.739130</td>
      <td>7.75</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>2016-03-31</td>
      <td>3308</td>
      <td>964.0</td>
      <td>904.0</td>
      <td>904.0</td>
      <td>904.0</td>
      <td>904.0</td>
      <td>904.0</td>
      <td>904.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>959.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>49.0</td>
      <td>777</td>
      <td>944.652174</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>2016-03-31</td>
      <td>3373</td>
      <td>282.0</td>
      <td>310.0</td>
      <td>310.0</td>
      <td>310.0</td>
      <td>310.0</td>
      <td>310.0</td>
      <td>310.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>314.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>100.0</td>
      <td>3770</td>
      <td>312.956522</td>
      <td>3.25</td>
    </tr>
  </tbody>
</table>
<p>919 rows × 39 columns</p>
</div>


### Data Aggregation:
 
As we can see in the above dataframe, we have an hourly ranking of applications on a daily basis. An important thing to note is that these ranks are deviating within a given short range of values. In such cases, we can also take the average value of all the ranks and ratings to simplify our problem and then create a single column for both the properties.


```python
subsetDataFrame['avgrank'] = subsetDataFrame.iloc[:,3:26].mean(axis=1)
subsetDataFrame['avgrating'] = subsetDataFrame.iloc[:,27:31].mean(axis=1)

```


```python
newData = subsetDataFrame[['Date','app_id','sales','avgrank','avgrating']]
```

### Data Extraction:

We will now only consider the average rank & rating columns instead of hourly rank columns. For this, we subsetted the data and extracted only the relevant columns in a new data frame.


```python
print(newData)
```

                Date  app_id  sales     avgrank  avgrating
    0     2016-01-01     320   2412  469.739130       0.50
    1     2016-01-01     406   1308  719.173913       0.00
    2     2016-01-01     459   2037  603.695652       0.25
    3     2016-01-01     722   2052  567.913043       2.75
    4     2016-01-01    1234   1553  776.826087       0.00
    5     2016-01-01    1490   2152  518.260870       0.50
    7     2016-01-01    2891  89289    5.782609      33.50
    8     2016-01-02     320   1622  464.043478       0.00
    9     2016-01-02     406   1110  754.652174       0.00
    10    2016-01-02     459   1553  586.217391       0.25
    11    2016-01-02     722   1467  617.782609       1.00
    12    2016-01-02    1234   1011  727.695652       0.00
    13    2016-01-02    1490   1722  546.260870       0.50
    16    2016-01-02    2891  84915    5.826087      35.25
    17    2016-01-03     320   1564  508.695652       0.00
    18    2016-01-03     406    963  762.434783       0.00
    19    2016-01-03     459   1409  560.521739       0.00
    20    2016-01-03     722   1371  676.000000       0.50
    21    2016-01-03    1234    938  829.217391       0.00
    22    2016-01-03    1490   1662  560.913043       0.00
    23    2016-01-03    2346   2029  680.086957       0.00
    24    2016-01-03    2891  61345    5.000000      24.25
    25    2016-01-04     320   1609  525.695652       0.00
    26    2016-01-04     406   1168  797.217391       0.00
    27    2016-01-04     459   1732  573.826087       0.00
    28    2016-01-04     722   1421  715.043478       0.50
    29    2016-01-04    1234    989  840.652174       0.00
    30    2016-01-04    1490   1081  573.782609       1.00
    31    2016-01-04    2346   1460  654.565217       0.25
    32    2016-01-04    2891  58943    6.391304      24.75
    33    2016-01-05     320   1594  548.173913       0.00
    34    2016-01-05     406   1163  750.869565       0.00
    35    2016-01-05     459   1392  563.130435       0.25
    36    2016-01-05     722   1482  757.086957       1.00
    37    2016-01-05    1234   1055  875.173913       0.25
    38    2016-01-05    1490    984  743.217391       0.00
    39    2016-01-05    2346   1102  807.869565       0.50
    40    2016-01-05    2891  62196   10.782609      11.25
    41    2016-01-06     320   1883  545.739130       0.00
    42    2016-01-06     406   1252  742.173913       0.00
    44    2016-01-06     722   1737  706.608696       1.00
    45    2016-01-06    1234   1227  855.782609       0.00
    46    2016-01-06    1490   1244  848.260870       1.25
    49    2016-01-06    2891  67759   10.478261      14.50
    50    2016-01-07     320   1978  527.478261       0.00
    51    2016-01-07     406   1254  763.217391       0.00
    53    2016-01-07     722   1538  701.173913       1.25
    54    2016-01-07    1234   1125  845.086957       0.00
    55    2016-01-07    1490   1184  878.869565       0.25
    56    2016-01-07    2398    956  940.260870       0.00
    57    2016-01-07    2891  60462    9.434783      13.25
    58    2016-01-08     320   1509  483.608696       0.00
    59    2016-01-08     406    960  736.043478       0.00
    61    2016-01-08     722   1273  702.695652       0.50
    62    2016-01-08    1234    893  818.695652       0.00
    63    2016-01-08    1490    876  899.391304       0.00
    65    2016-01-08    2891  36569   11.608696      10.00
    66    2016-01-09     320   1256  502.652174       0.00
    67    2016-01-09     406    761  783.478261       0.00
    68    2016-01-09     459    867  657.913043       0.25
    69    2016-01-09     722   1038  705.956522       2.00
    70    2016-01-09    1234    808  789.695652       0.00
    71    2016-01-09    1490    871  872.608696       0.50
    72    2016-01-09    2891  28872   16.695652       4.25
    73    2016-01-10     320   1316  519.652174       0.25
    74    2016-01-10     406    775  815.913043       0.00
    76    2016-01-10     722   1248  702.260870       0.00
    78    2016-01-10    1234    795  804.869565       0.50
    79    2016-01-10    1490    866  844.000000       0.00
    80    2016-01-10    2891  32229   20.608696       6.50
    81    2016-01-11     320   1350  524.173913       0.00
    82    2016-01-11     406    744  832.000000       0.00
    83    2016-01-11     459   1454  430.869565       0.00
    84    2016-01-11     722   1387  651.565217       1.50
    85    2016-01-11     907   1008  857.695652       0.00
    86    2016-01-11    1234    879  824.782609       0.00
    87    2016-01-11    1490   1505  811.608696       0.75
    90    2016-01-11    2891  30459   19.086957       5.25
    91    2016-01-12     320   1650  525.478261       0.00
    92    2016-01-12     406   1071  867.130435       0.00
    93    2016-01-12     459   1589  481.826087       0.50
    94    2016-01-12     722   1622  588.739130       1.50
    95    2016-01-12     907    909  923.565217       0.00
    96    2016-01-12    1234    946  821.347826       0.25
    97    2016-01-12    1490   2339  675.521739       0.25
    98    2016-01-12    2346   2047  774.913043       0.25
    100   2016-01-12    2891  30254   21.608696       4.50
    122   2016-01-14     320   2046  562.434783       0.00
    123   2016-01-14     406   1285  824.391304       0.00
    124   2016-01-14     459   1893  480.826087       0.00
    129   2016-01-14    1490   2371  636.304348       0.25
    134   2016-01-14    2891  49142   19.782609       8.25
    135   2016-01-15     320   1813  531.565217       0.25
    136   2016-01-15     406   1132  843.869565       0.00
    137   2016-01-15     459   1478  556.695652       0.00
    138   2016-01-15     722   1613  686.086957       0.50
    139   2016-01-15    1234   1110  810.347826       0.00
    140   2016-01-15    1490   2081  637.739130       0.00
    141   2016-01-15    2346   1769  728.521739       0.00
    142   2016-01-15    2891  37157   19.173913      10.75
    143   2016-01-16     320   1372  493.043478       0.00
    144   2016-01-16     406    844  834.304348       0.00
    146   2016-01-16     722   1391  685.695652       1.25
    147   2016-01-16    1234    900  826.260870       0.00
    148   2016-01-16    1490   1554  639.260870       0.25
    149   2016-01-16    2346   1443  757.043478       0.00
    150   2016-01-16    2891  30890   23.956522       4.25
    151   2016-01-17     320   1385  519.478261       0.00
    152   2016-01-17     406    892  823.521739       0.00
    153   2016-01-17     459    977  624.521739       0.50
    154   2016-01-17     722   2729  606.086957       1.25
    155   2016-01-17    1234    919  828.739130       0.25
    156   2016-01-17    1490   1673  661.000000       1.00
    157   2016-01-17    2346   1115  787.478261       1.00
    159   2016-01-17    2891  27467   25.695652       5.50
    160   2016-01-18     320   1291  546.608696       0.25
    161   2016-01-18     406    777  841.695652       0.00
    163   2016-01-18     722   3084  433.347826       0.75
    164   2016-01-18    1234    804  809.391304       0.00
    165   2016-01-18    1490    943  678.217391       0.50
    167   2016-01-18    2667  44326   68.956522      10.50
    168   2016-01-18    2891  22792   28.304348       3.75
    169   2016-01-19     320   1368  545.565217       0.00
    170   2016-01-19     406    945  879.913043       0.00
    171   2016-01-19     459   1977  733.652174       0.50
    172   2016-01-19     722   2754  383.173913       1.00
    173   2016-01-19    1234    920  835.869565       0.00
    174   2016-01-19    1490   1094  788.434783       0.25
    176   2016-01-19    2667  47512   25.739130       6.75
    177   2016-01-19    2891  29160   32.608696       5.75
    178   2016-01-20     320   1746  565.173913       0.00
    179   2016-01-20     406   1267  876.347826       0.00
    181   2016-01-20     722   3107  390.652174       0.75
    182   2016-01-20    1234   1096  859.695652       0.00
    183   2016-01-20    1490   1319  749.478261       0.50
    184   2016-01-20    2346   1355  766.130435       0.50
    186   2016-01-20    2667  49850   24.826087      12.50
    187   2016-01-20    2891  36121   33.000000       8.25
    188   2016-01-21     320   1828  555.695652       0.00
    189   2016-01-21     406   1124  805.956522       0.00
    190   2016-01-21     459   1923  401.782609       0.25
    191   2016-01-21     722   2897  407.304348       1.25
    192   2016-01-21    1234   1220  848.739130       0.25
    193   2016-01-21    1490   1399  714.043478       0.00
    194   2016-01-21    2346   1582  854.217391       0.00
    195   2016-01-21    2490   1958  838.695652       0.25
    196   2016-01-21    2667  45371   27.652174       8.25
    197   2016-01-21    2891  34248   35.478261       7.50
    198   2016-01-22     320   1232  514.913043       0.25
    199   2016-01-22     406    727  821.478261       0.00
    200   2016-01-22     459   1532  462.260870       0.25
    201   2016-01-22     722   2090  431.043478       0.75
    202   2016-01-22    1234    842  768.130435       0.00
    203   2016-01-22    1490   1043  692.086957       0.50
    204   2016-01-22    2346   1284  892.043478       0.25
    205   2016-01-22    2490   1654  667.043478       0.50
    206   2016-01-22    2667  28863   29.000000       6.75
    207   2016-01-22    2891  21218   35.913043       6.00
    208   2016-01-23     320   1167  502.173913       0.00
    209   2016-01-23     406    798  847.739130       0.00
    210   2016-01-23     459   1198  479.521739       0.25
    211   2016-01-23     722   2135  433.043478       1.00
    212   2016-01-23    1234    775  769.173913       0.25
    213   2016-01-23    1490    856  679.826087       0.00
    214   2016-01-23    2346   1276  794.826087       0.25
    215   2016-01-23    2490   1192  629.391304       0.25
    218   2016-01-24     320   1189  532.695652       0.00
    219   2016-01-24     406    708  862.695652       0.00
    220   2016-01-24     459   2028  474.652174       0.00
    221   2016-01-24     722   1781  450.130435       3.75
    223   2016-01-24    1234    794  802.521739       0.00
    224   2016-01-24    1490    886  737.347826       0.25
    225   2016-01-24    2346   1120  750.782609       0.75
    226   2016-01-24    2490   1168  645.434783       0.00
    227   2016-01-24    2667  21527   36.782609       6.75
    228   2016-01-24    2891  18949   32.826087       4.50
    229   2016-01-25     320   1250  539.304348       0.00
    230   2016-01-25     406    779  894.130435       0.00
    231   2016-01-25     459   1336  423.304348       0.25
    232   2016-01-25     722   1703  501.217391       4.00
    233   2016-01-25     907   1019  928.956522       0.00
    234   2016-01-25    1234    791  870.000000       0.25
    235   2016-01-25    1490    935  732.086957       0.00
    236   2016-01-25    2346   1087  811.565217       0.25
    237   2016-01-25    2490   1259  726.565217       0.50
    238   2016-01-25    2667  16278   46.739130       4.25
    239   2016-01-25    2891  20721   36.869565       2.75
    240   2016-01-26     320   1368  545.782609       0.00
    241   2016-01-26     406    924  902.130435       0.00
    242   2016-01-26     459   1431  520.347826       0.00
    243   2016-01-26     722   2316  507.782609       1.75
    244   2016-01-26     907    949  932.826087       0.50
    245   2016-01-26    1234    931  843.956522       0.25
    246   2016-01-26    1490   1029  723.521739       0.00
    248   2016-01-26    2490   1204  702.347826       0.00
    249   2016-01-26    2667  16522   63.478261       4.00
    250   2016-01-26    2891  27454   33.217391       3.50
    251   2016-01-27     320   1807  553.434783       0.00
    252   2016-01-27     406   1317  874.086957       0.00
    253   2016-01-27     459   2079  520.347826       0.50
    254   2016-01-27     722   2487  482.565217       3.25
    256   2016-01-27    1234   1157  853.130435       0.00
    257   2016-01-27    1490   1213  757.086957       0.25
    259   2016-01-27    2490   1758  746.652174       0.50
    260   2016-01-27    2667  18981   77.739130       3.75
    261   2016-01-27    2891  34159   31.391304       6.75
    262   2016-01-28     320   1639  554.608696       0.25
    263   2016-01-28     406   1188  799.391304       0.00
    264   2016-01-28     459   2361  514.521739       0.25
    265   2016-01-28     722   2116  505.304348       1.00
    266   2016-01-28    1234   1147  840.000000       0.00
    267   2016-01-28    1490   1321  781.956522       0.75
    269   2016-01-28    2490   1935  726.347826       0.00
    270   2016-01-28    2667  17933   85.260870       3.00
    271   2016-01-28    2891  34537   30.826087       6.50
    272   2016-01-29     320   1240  537.173913       0.00
    273   2016-01-29     406    703  816.304348       0.00
    274   2016-01-29     459   1480  456.391304       0.25
    275   2016-01-29     722   1502  540.739130       2.50
    276   2016-01-29    1234    827  842.521739       0.25
    277   2016-01-29    1490    863  748.565217       0.75
    278   2016-01-29    2398   7055  164.739130       0.00
    279   2016-01-29    2490    386  736.652174       0.25
    280   2016-01-29    2667  13235   88.956522       3.25
    281   2016-01-29    2891  20230   30.913043       3.00
    282   2016-01-30     320   1175  529.652174       0.00
    283   2016-01-30     406    667  868.956522       0.00
    284   2016-01-30     459   1409  467.521739       0.00
    285   2016-01-30     722   1727  564.826087       2.75
    286   2016-01-30    1234    797  824.173913       0.00
    287   2016-01-30    1490   1197  789.565217       0.75
    289   2016-01-30    2398   7829  138.565217       0.25
    291   2016-01-30    2667  13021   87.304348       1.75
    292   2016-01-30    2891  16766   38.478261       4.75
    294   2016-01-31     320   1149  532.739130       0.25
    295   2016-01-31     406    711  896.478261       0.00
    296   2016-01-31     459   1207  495.521739       0.25
    297   2016-01-31     722   2059  475.086957       6.75
    298   2016-01-31    1234    790  781.086957       0.00
    299   2016-01-31    1490    739  676.043478       0.50
    301   2016-01-31    2398   6435  128.565217       0.00
    302   2016-01-31    2667  11656   91.043478       4.75
    303   2016-01-31    2891  15648   45.173913       4.25
    304   2016-02-01     320   1270  547.086957       0.25
    305   2016-02-01     406    773  884.434783       0.00
    306   2016-02-01     459   1271  525.869565       0.00
    307   2016-02-01     722   1649  473.521739       1.50
    309   2016-02-01    1234    928  777.652174       0.00
    310   2016-02-01    1490    751  772.565217       0.00
    312   2016-02-01    2398   6630  134.260870       0.00
    313   2016-02-01    2667   8485  105.260870       1.75
    314   2016-02-01    2891  16267   50.565217       4.50
    316   2016-02-02     320   1435  549.434783       0.00
    317   2016-02-02     406   1003  873.434783       0.00
    318   2016-02-02     459   1181  588.391304       0.25
    319   2016-02-02     722   1770  505.000000       0.50
    320   2016-02-02    1234   1080  747.652174       0.25
    323   2016-02-02    2398   4731  155.739130       0.25
    324   2016-02-02    2667   8677  145.565217       1.50
    325   2016-02-02    2891  18581   56.521739       3.75
    327   2016-02-03     320   1858  563.478261       0.00
    328   2016-02-03     406   1419  817.521739       0.00
    329   2016-02-03     459   1409  630.000000       0.25
    330   2016-02-03     722   2330  540.521739       2.50
    331   2016-02-03    1234   1361  768.043478       0.00
    333   2016-02-03    2346   2569  765.043478       0.00
    334   2016-02-03    2398   6081  197.130435       0.50
    335   2016-02-03    2667   9948  163.565217       1.25
    336   2016-02-03    2891  25123   57.173913       5.00
    339   2016-02-04     320   1869  537.782609       0.00
    340   2016-02-04     406   1312  813.956522       0.00
    341   2016-02-04     459   1678  648.086957       0.00
    342   2016-02-04     722   2164  550.304348       1.50
    343   2016-02-04    1234   1254  737.217391       0.00
    345   2016-02-04    2346   3304  621.826087       0.00
    346   2016-02-04    2398   4468  228.086957       0.00
    347   2016-02-04    2667   9212  183.956522       3.00
    348   2016-02-04    2891  23394   59.565217       7.75
    349   2016-02-04    3082   6235  262.086957       1.25
    351   2016-02-05     320   1320  520.086957       0.25
    352   2016-02-05     406    846  800.695652       0.00
    353   2016-02-05     459   1293  616.869565       0.00
    354   2016-02-05     722   1449  561.782609       0.50
    355   2016-02-05    1234    925  754.173913       0.25
    356   2016-02-05    2346   2616  541.304348       0.00
    357   2016-02-05    2398   2302  293.695652       0.00
    358   2016-02-05    2667   6666  186.130435       2.00
    359   2016-02-05    2891  14004   60.043478       3.50
    360   2016-02-05    3082   3760  248.391304       0.75
    362   2016-02-06     320   1280  533.695652       0.00
    363   2016-02-06     406    769  808.304348       0.00
    364   2016-02-06     459   1014  612.695652       0.25
    365   2016-02-06     722   1769  557.695652       1.25
    366   2016-02-06    1234    925  759.521739       0.00
    367   2016-02-06    2346   2343  530.391304       0.00
    368   2016-02-06    2398   1973  386.956522       0.00
    369   2016-02-06    2667   6154  177.565217       9.25
    370   2016-02-06    2891  13272   65.478261       3.00
    371   2016-02-06    3082   3408  295.260870       1.00
    372   2016-02-06    3333   3552  364.260870       0.75
    373   2016-02-07     320   1439  536.434783       0.00
    374   2016-02-07     406    929  850.434783       0.00
    375   2016-02-07     459    979  654.217391       0.50
    376   2016-02-07     722   1663  515.086957       1.25
    377   2016-02-07    1234    989  789.565217       0.25
    378   2016-02-07    2346   2189  521.739130       0.50
    379   2016-02-07    2398   1938  458.608696       0.00
    380   2016-02-07    2667   6039  181.217391       7.50
    381   2016-02-07    2891  12973   69.869565       2.00
    382   2016-02-07    3082   3955  334.391304       1.75
    383   2016-02-07    3333   3062  379.000000       0.00
    384   2016-02-08     320   1359  530.869565       0.00
    385   2016-02-08     406    815  838.304348       0.00
    386   2016-02-08     459   1439  692.521739       0.25
    387   2016-02-08     722   1667  554.956522       1.25
    388   2016-02-08    1234    927  768.913043       0.00
    389   2016-02-08    2346   2127  516.391304       0.00
    390   2016-02-08    2398   1618  515.260870       0.00
    391   2016-02-08    2667   5211  193.304348       2.50
    392   2016-02-08    2891  11992   78.173913       2.25
    393   2016-02-08    3082   1317  335.000000       0.75
    394   2016-02-08    3333   2002  435.304348       0.00
    395   2016-02-09     320   1735  521.478261       0.00
    396   2016-02-09     406   1020  816.739130       0.00
    397   2016-02-09     459   1465  580.739130       0.00
    398   2016-02-09     722   2006  607.391304       1.00
    400   2016-02-09    1234   1088  752.304348       0.00
    401   2016-02-09    2346   2027  548.434783       0.00
    402   2016-02-09    2398   1568  558.826087       0.00
    403   2016-02-09    2667   5879  210.043478       3.00
    404   2016-02-09    2891  15179   78.130435       3.00
    405   2016-02-09    3082    556  606.913043       0.75
    406   2016-02-09    3333   2652  537.695652       0.00
    407   2016-02-10     320   2212  527.739130       0.25
    408   2016-02-10     406   1385  764.434783       0.00
    409   2016-02-10     459   1697  581.565217       0.50
    410   2016-02-10     722   2673  575.869565       1.75
    411   2016-02-10    1234   1448  762.478261       0.00
    412   2016-02-10    2346   3024  627.956522       0.25
    413   2016-02-10    2398   1736  650.608696       0.00
    414   2016-02-10    2667   6547  229.260870       3.75
    415   2016-02-10    2891  20504   80.869565       4.50
    418   2016-02-10    3333   2061  569.391304       0.25
    419   2016-02-11     320   2064  481.347826       0.00
    420   2016-02-11     406   1238  785.565217       0.00
    421   2016-02-11     459   1760  622.608696       0.00
    422   2016-02-11     722   2236  552.000000       1.25
    423   2016-02-11    1234   1381  714.173913       0.00
    424   2016-02-11    2346   3513  602.695652       0.00
    425   2016-02-11    2398   1719  707.347826       0.00
    426   2016-02-11    2667   6110  261.956522       4.50
    427   2016-02-11    2891  17485   76.695652       5.00
    429   2016-02-11    3333   1916  794.652174       0.25
    430   2016-02-12     320   1449  467.000000       0.00
    431   2016-02-12     406    791  825.173913       0.00
    432   2016-02-12     459   1235  604.130435       0.50
    433   2016-02-12     722   1489  593.043478       0.50
    434   2016-02-12    1234   1001  709.000000       0.00
    435   2016-02-12    2346   2534  523.347826       1.00
    436   2016-02-12    2398   1232  723.000000       0.00
    437   2016-02-12    2667   4729  272.521739       1.50
    438   2016-02-12    2891  11147   79.086957       4.25
    442   2016-02-13     320   1342  487.478261       0.00
    443   2016-02-13     406    702  858.956522       0.00
    444   2016-02-13     459   1728  563.434783       0.50
    445   2016-02-13     722   1380  606.695652       1.00
    447   2016-02-13    1234   1009  678.956522       0.00
    448   2016-02-13    2346   2332  462.000000       0.75
    450   2016-02-13    2398   1271  739.304348       0.25
    451   2016-02-13    2667   4395  261.391304       2.50
    452   2016-02-13    2891   9598   82.260870       2.50
    454   2016-02-14      82   2928  223.304348       0.00
    455   2016-02-14     320   1280  506.347826       0.00
    456   2016-02-14     406    674  904.260870       0.00
    457   2016-02-14     459    949  473.565217       0.25
    458   2016-02-14     722   1353  625.956522       0.25
    459   2016-02-14     907    746  924.086957       0.25
    460   2016-02-14    1234    877  684.695652       0.25
    461   2016-02-14    2346   1954  451.130435       0.25
    462   2016-02-14    2373    826  742.739130       1.00
    463   2016-02-14    2398   1090  701.347826       0.00
    464   2016-02-14    2667   4079  261.043478       3.00
    465   2016-02-14    2891  10271   87.956522       3.50
    468   2016-02-15      82   2029  278.826087       0.25
    469   2016-02-15     320   1430  485.391304       0.00
    470   2016-02-15     406    768  902.086957       0.00
    471   2016-02-15     459    940  562.173913       0.00
    472   2016-02-15     722   1441  606.608696       1.25
    474   2016-02-15    1234    969  692.000000       0.00
    476   2016-02-15    2346   2134  479.173913       0.25
    478   2016-02-15    2398   1316  726.478261       0.00
    479   2016-02-15    2667   4924  253.956522       2.25
    480   2016-02-15    2891  11795   78.000000       3.75
    483   2016-02-16      82   1851  415.913043       0.25
    484   2016-02-16     320   1582  480.956522       0.00
    485   2016-02-16     406   1000  888.434783       0.00
    486   2016-02-16     459   1409  644.608696       0.00
    487   2016-02-16     722   1997  598.000000       0.50
    488   2016-02-16    1234   1079  726.913043       0.25
    489   2016-02-16    1490   1562  867.652174       6.75
    490   2016-02-16    2346   2062  503.173913       0.00
    491   2016-02-16    2398   1478  710.826087       0.00
    492   2016-02-16    2667   5353  251.782609       2.75
    493   2016-02-16    2891  15537   75.478261       5.00
    494   2016-02-17      82   1723  561.956522       0.25
    495   2016-02-17     320   1919  503.304348       0.00
    496   2016-02-17     406   1307  865.000000       0.00
    497   2016-02-17     459   1661  582.913043       0.25
    498   2016-02-17     722   2217  562.391304       0.75
    499   2016-02-17    1234   1327  750.565217       0.00
    500   2016-02-17    1490   1960  807.521739       2.25
    501   2016-02-17    2346   2706  569.739130       0.00
    502   2016-02-17    2398   1865  747.043478       0.25
    503   2016-02-17    2667   8867  265.608696       1.50
    504   2016-02-17    2891  20417   74.652174       5.75
    506   2016-02-18      82   1419  705.086957       0.00
    507   2016-02-18     320   2025  506.782609       0.00
    508   2016-02-18     406   1293  802.217391       0.00
    509   2016-02-18     459   1821  584.217391       0.25
    510   2016-02-18     722   2385  574.869565       0.75
    511   2016-02-18    1234   1310  771.304348       0.00
    512   2016-02-18    1490   1734  835.739130       1.50
    513   2016-02-18    2346   3556  552.391304       0.00
    514   2016-02-18    2398   2286  733.695652       0.25
    515   2016-02-18    2667  35279  184.565217       5.00
    516   2016-02-18    2891  20284   72.043478       7.00
    518   2016-02-19      82   1357  798.869565       0.00
    519   2016-02-19     320   1560  505.173913       0.00
    520   2016-02-19     406   1021  810.000000       0.00
    521   2016-02-19     459    715  628.000000       0.25
    522   2016-02-19     722   1969  533.086957       1.75
    523   2016-02-19    1234   1170  775.478261       0.25
    524   2016-02-19    1490   1476  859.521739       3.00
    525   2016-02-19    2346   3099  496.043478       0.50
    526   2016-02-19    2398   1933  668.304348       0.00
    527   2016-02-19    2667  16330   78.826087       2.75
    528   2016-02-19    2891  19032   71.130435       4.75
    531   2016-02-20     320   1302  513.260870       0.00
    532   2016-02-20     406    856  844.695652       0.00
    533   2016-02-20     459   1271  794.260870       0.00
    534   2016-02-20     722   1555  542.043478       1.25
    535   2016-02-20    1234    946  780.913043       0.25
    536   2016-02-20    1490   1163  839.000000       1.00
    537   2016-02-20    2346   3295  443.869565       0.25
    538   2016-02-20    2398   1310  670.000000       0.00
    539   2016-02-20    2667  10265   98.652174       2.00
    540   2016-02-20    2891  14375   64.173913       6.00
    542   2016-02-21     320   1357  508.869565       0.25
    543   2016-02-21     406    799  826.521739       0.00
    544   2016-02-21     459   1187  679.913043       0.00
    545   2016-02-21     722   1413  560.130435       0.50
    546   2016-02-21    1234    972  771.695652       0.00
    547   2016-02-21    1490   1138  880.826087       0.25
    548   2016-02-21    2346   3777  381.652174       0.50
    550   2016-02-21    2398   1256  706.695652       0.00
    551   2016-02-21    2667   8114  119.260870       2.25
    552   2016-02-21    2891  14902   63.478261       3.00
    554   2016-02-22     320   1396  508.086957       0.00
    555   2016-02-22     406    789  821.217391       0.00
    556   2016-02-22     459   1086  623.347826       0.25
    557   2016-02-22     722   1486  600.391304       1.00
    558   2016-02-22    1234   1024  735.217391       0.25
    559   2016-02-22    1490    806  882.956522       0.00
    560   2016-02-22    2346   5447  321.782609       0.00
    561   2016-02-22    2373    820  657.739130       1.00
    562   2016-02-22    2398   1297  741.391304       0.00
    563   2016-02-22    2667   7352  150.478261       2.75
    564   2016-02-22    2891  18591   60.608696       2.75
    567   2016-02-23     320   1654  499.739130       0.00
    568   2016-02-23     406   1059  848.130435       0.25
    569   2016-02-23     459   1381  589.217391       0.00
    570   2016-02-23     722   1763  610.130435       0.25
    572   2016-02-23    1234   1149  717.173913       0.00
    574   2016-02-23    2346   3896  349.652174       0.00
    576   2016-02-23    2398   1469  736.043478       0.00
    577   2016-02-23    2667   7706  163.869565       1.75
    578   2016-02-23    2891  21589   45.000000       4.00
    581   2016-02-24     320   2128  492.913043       0.00
    582   2016-02-24     406   1373  777.217391       0.00
    583   2016-02-24     459   1614  581.608696       1.00
    585   2016-02-24     722   2165  589.608696       2.00
    586   2016-02-24    1234   1373  706.869565       0.00
    588   2016-02-24    2346   3034  431.347826       0.50
    589   2016-02-24    2398   2178  718.782609       0.25
    590   2016-02-24    2667   8076  187.347826       3.00
    592   2016-02-24    2891  29501   40.173913       5.50
    593   2016-02-25     320   1936  477.739130       0.00
    594   2016-02-25     406   1227  760.391304       0.00
    595   2016-02-25     459   1717  638.000000       0.50
    596   2016-02-25     676    983  933.434783       0.00
    597   2016-02-25     722   1981  595.086957       1.50
    598   2016-02-25    1234   1319  734.652174       0.00
    600   2016-02-25    2346   3048  529.739130       0.00
    601   2016-02-25    2398   1689  693.478261       0.00
    602   2016-02-25    2667   5546  221.956522       1.75
    603   2016-02-25    2821   1647  619.304348       0.25
    604   2016-02-25    2891  31090   35.304348      10.00
    605   2016-02-26     320   1480  468.173913       0.00
    606   2016-02-26     406    811  774.130435       0.00
    607   2016-02-26     459   1055  620.739130       0.25
    608   2016-02-26     676    627  925.173913       0.00
    609   2016-02-26     722   1183  612.086957       0.50
    610   2016-02-26    1234    962  725.130435       0.00
    612   2016-02-26    2346   2712  509.913043       0.00
    613   2016-02-26    2398   1020  762.695652       0.00
    614   2016-02-26    2667   4138  266.304348       2.25
    615   2016-02-26    2821    176  648.521739       0.25
    616   2016-02-26    2891  16154   34.521739       5.00
    618   2016-02-27     320   1458  422.956522       0.00
    619   2016-02-27     406    700  785.695652       0.25
    620   2016-02-27     459    865  646.869565       0.00
    621   2016-02-27     676    563  934.956522       0.00
    622   2016-02-27     722   1314  668.826087       3.00
    624   2016-02-27    1234    892  696.913043       0.25
    625   2016-02-27    2346   2770  426.565217       0.75
    626   2016-02-27    2398   1199  781.130435       0.00
    627   2016-02-27    2667   4485  271.695652       1.00
    629   2016-02-27    2891  15148   41.130435       5.25
    632   2016-02-28     320   1430  414.217391       0.25
    633   2016-02-28     406    768  803.000000       0.00
    634   2016-02-28     459   1364  670.347826       0.00
    636   2016-02-28     722   1411  618.869565       5.75
    638   2016-02-28    1234    951  708.826087       0.00
    639   2016-02-28    2346   2600  374.608696       0.50
    640   2016-02-28    2398   1438  758.260870       0.25
    641   2016-02-28    2667   4460  260.695652       1.75
    642   2016-02-28    2891  15738   43.478261       5.75
    643   2016-02-28    3308   1047  880.260870       0.25
    645   2016-03-01     320   1586  423.913043       0.25
    646   2016-03-01     406    853  796.217391       0.00
    647   2016-03-01     459   1590  560.869565       0.50
    648   2016-03-01     676    662  957.086957       0.00
    649   2016-03-01     722   3075  571.608696       2.50
    651   2016-03-01    1234    979  683.739130       0.00
    653   2016-03-01    2346   1548  398.913043       0.50
    654   2016-03-01    2398   1401  706.304348       0.00
    655   2016-03-01    2667   5233  245.565217       2.75
    656   2016-03-01    2891  19151   39.782609       2.00
    657   2016-03-01    3308   1200  745.434783       0.50
    659   2016-03-02     320   2041  417.260870       0.00
    660   2016-03-02     406   1012  813.565217       0.00
    661   2016-03-02     459   1786  506.043478       0.25
    662   2016-03-02     676    823  928.217391       0.00
    663   2016-03-02     722   4436  431.434783       0.75
    664   2016-03-02    1234   1114  681.217391       0.00
    667   2016-03-02    2333   4691  477.260870       0.25
    668   2016-03-02    2346   1542  537.043478       0.00
    669   2016-03-02    2398   2221  635.086957       0.00
    670   2016-03-02    2667   5296  223.304348       1.50
    671   2016-03-02    2891  25378   34.826087       4.50
    672   2016-03-02    3308   1252  663.565217       0.00
    673   2016-03-03     320   2322  393.304348       0.25
    674   2016-03-03     406   1277  805.869565       0.00
    675   2016-03-03     459   1508  524.391304       0.25
    676   2016-03-03     676   1049  924.826087       0.00
    677   2016-03-03     722   5262  371.695652       1.50
    678   2016-03-03    1234   1294  716.826087       0.00
    679   2016-03-03    1490   1317  872.565217       0.75
    680   2016-03-03    2333   6659  327.434783       0.00
    681   2016-03-03    2346   1706  683.217391       0.25
    682   2016-03-03    2398   2198  505.304348       0.00
    683   2016-03-03    2667   5575  253.652174       2.25
    684   2016-03-03    2891  33408   29.565217       4.00
    685   2016-03-03    3308   1982  662.434783       0.25
    687   2016-03-04     320   2084  404.826087       0.00
    688   2016-03-04     406   1123  801.304348       0.00
    689   2016-03-04     459   1720  594.521739       0.50
    690   2016-03-04     676    990  915.913043       0.00
    691   2016-03-04     722   4766  364.130435       1.50
    692   2016-03-04    1234   1396  752.217391       0.00
    693   2016-03-04    1490   1115  904.434783       0.75
    694   2016-03-04    2333   6417  253.739130       0.25
    695   2016-03-04    2346   1925  776.739130       0.00
    696   2016-03-04    2398   1691  573.000000       0.25
    698   2016-03-04    2667   5401  291.782609       4.25
    699   2016-03-04    2891  32011   30.565217       6.25
    700   2016-03-04    3308   2568  612.521739       0.50
    701   2016-03-05     320   1581  405.043478       0.00
    702   2016-03-05     406    664  823.173913       0.25
    703   2016-03-05     459   1888  566.000000       0.00
    704   2016-03-05     676    631  892.000000       0.00
    705   2016-03-05     722   3108  368.608696       1.75
    707   2016-03-05    1234   3348  572.695652       0.00
    708   2016-03-05    1490   1131  890.869565       0.00
    709   2016-03-05    2333   4743  255.695652       0.50
    710   2016-03-05    2346   1862  716.565217       0.00
    711   2016-03-05    2398    937  695.913043       0.00
    713   2016-03-05    2667   3862  293.869565       0.75
    714   2016-03-05    2891  32532   28.869565       2.25
    715   2016-03-05    3308   2426  496.521739       0.25
    717   2016-03-06     320   1527  410.434783       0.00
    718   2016-03-06     406    757  886.260870       0.00
    719   2016-03-06     459   1526  437.260870       0.25
    720   2016-03-06     676    786  832.913043       0.00
    721   2016-03-06     722   3066  382.000000       1.25
    722   2016-03-06     907    928  922.652174       0.00
    723   2016-03-06    1234   1710  350.260870       0.00
    724   2016-03-06    1490   1391  779.956522       1.25
    726   2016-03-06    2333   2689  246.260870       1.00
    727   2016-03-06    2346   1827  603.695652       0.25
    728   2016-03-06    2398    990  778.913043       0.00
    729   2016-03-06    2667   4908  289.695652       1.00
    730   2016-03-06    2891  33458   17.130435       4.00
    731   2016-03-06    3308   1779  412.652174       0.25
    733   2016-03-07     320   1603  431.521739       0.00
    734   2016-03-07     406    798  928.130435       0.00
    735   2016-03-07     459   1207  496.043478       0.00
    736   2016-03-07     676   1008  780.260870       0.00
    737   2016-03-07     722   3151  398.304348       0.75
    739   2016-03-07    1234   1368  425.695652       0.00
    740   2016-03-07    1490   1529  659.130435       1.50
    741   2016-03-07    2333   1731  354.260870       1.00
    742   2016-03-07    2346   2077  595.000000       0.25
    743   2016-03-07    2398   1188  812.000000       0.00
    744   2016-03-07    2667   4436  257.869565       1.00
    745   2016-03-07    2891  34919   14.521739       4.50
    746   2016-03-07    3308   1640  447.478261       0.50
    747   2016-03-08     320   1469  422.608696       0.00
    748   2016-03-08     406    711  898.434783       0.00
    749   2016-03-08     459   1096  612.956522       0.00
    750   2016-03-08     676    757  702.347826       0.00
    751   2016-03-08     722   2009  415.086957       0.25
    752   2016-03-08    1234   1082  553.043478       0.00
    753   2016-03-08    1490   1259  593.217391       0.50
    754   2016-03-08    2333   2015  562.565217       0.00
    755   2016-03-08    2346   2057  536.130435       0.00
    756   2016-03-08    2398   1143  768.130435       0.00
    757   2016-03-08    2667   3709  267.260870       0.50
    758   2016-03-08    2891  33293   13.652174       5.50
    759   2016-03-08    3308   1551  504.565217       0.50
    760   2016-03-09     320   1639  436.086957       0.25
    761   2016-03-09     406    957  930.086957       0.00
    762   2016-03-09     459   1567  623.260870       0.50
    763   2016-03-09     676    859  779.782609       0.00
    764   2016-03-09     722   1851  507.869565       0.50
    765   2016-03-09    1234   1259  628.869565       0.25
    767   2016-03-09    1490   1520  613.260870       0.75
    769   2016-03-09    2333   2084  595.000000       0.50
    770   2016-03-09    2346   1915  497.652174       0.00
    771   2016-03-09    2398   1290  829.478261       0.00
    772   2016-03-09    2667   3816  303.826087       0.75
    773   2016-03-09    2891  35441   13.565217       6.25
    774   2016-03-09    3308   2183  473.173913       0.75
    775   2016-03-10     320   1919  475.391304       0.00
    776   2016-03-10     406   1280  902.000000       0.00
    777   2016-03-10     459   1869  532.521739       0.25
    778   2016-03-10     676   1209  801.130435       0.00
    779   2016-03-10     722   2006  518.043478       1.00
    780   2016-03-10    1234   1477  645.000000       0.00
    781   2016-03-10    1461    998  877.217391       0.50
    782   2016-03-10    1490   1766  607.434783       0.50
    783   2016-03-10    1874  19010  135.608696       0.00
    784   2016-03-10    2333   2478  618.000000       0.25
    785   2016-03-10    2346   2616  571.478261       0.50
    786   2016-03-10    2398   1957  774.956522       0.00
    787   2016-03-10    2667   4539  333.391304       1.75
    788   2016-03-10    2891  41386   15.826087       6.00
    789   2016-03-10    3308   2946  408.347826       0.00
    805   2016-03-12     320   1395  481.565217       0.00
    806   2016-03-12     406    895  906.260870       0.00
    807   2016-03-12     459   1154  575.956522       0.25
    808   2016-03-12     676    919  752.565217       0.00
    809   2016-03-12     722   1231  650.043478       0.75
    810   2016-03-12    1234    874  730.173913       0.00
    812   2016-03-12    1490   1602  657.086957       1.25
    813   2016-03-12    1874  16656   45.130435       0.00
    814   2016-03-12    2333   1771  557.260870       0.50
    815   2016-03-12    2346   2661  457.869565       0.00
    816   2016-03-12    2398    993  873.608696       0.00
    817   2016-03-12    2667   3257  365.086957       0.75
    818   2016-03-12    2891  28069   23.000000       2.50
    819   2016-03-12    3308   1447  404.913043       0.00
    820   2016-03-13     320   1772  477.130435       0.00
    821   2016-03-13     406    833  891.086957       0.00
    822   2016-03-13     459   1127  615.434783       0.25
    823   2016-03-13     676   1036  747.739130       0.00
    824   2016-03-13     722   1188  691.826087       0.00
    825   2016-03-13    1234    898  764.565217       0.00
    826   2016-03-13    1461   2082  442.956522       0.75
    827   2016-03-13    1490   1502  599.130435       1.25
    828   2016-03-13    1874   4107   52.217391       0.25
    829   2016-03-13    2333   1385  595.260870       0.00
    830   2016-03-13    2346   2750  439.130435       0.00
    831   2016-03-13    2398   1027  931.608696       0.00
    832   2016-03-13    2667   2941  371.608696       1.25
    833   2016-03-13    2891  28991   22.826087       3.25
    834   2016-03-13    3308    872  531.130435       0.00
    835   2016-03-14     320   1528  431.565217       0.00
    836   2016-03-14     406    738  921.869565       0.00
    837   2016-03-14     459   1084  654.652174       0.00
    838   2016-03-14     676    871  783.043478       0.00
    839   2016-03-14     722   1163  734.521739       0.75
    840   2016-03-14    1234    880  771.608696       0.00
    841   2016-03-14    1461   1433  507.608696       1.50
    842   2016-03-14    1490   1039  607.478261       1.75
    844   2016-03-14    1874   1761  116.565217       0.00
    845   2016-03-14    2333   1777  683.217391       0.50
    846   2016-03-14    2346   2917  411.739130       0.50
    847   2016-03-14    2398    909  891.173913       0.00
    848   2016-03-14    2667   3078  405.478261       0.50
    849   2016-03-14    2891  27801   20.782609       4.50
    850   2016-03-14    3308    815  710.913043       0.00
    851   2016-03-15     320   1614  416.173913       0.25
    852   2016-03-15     406    758  931.260870       0.00
    853   2016-03-15     459    873  689.608696       0.00
    854   2016-03-15     676    743  778.521739       0.00
    855   2016-03-15     722   1139  751.695652       0.75
    856   2016-03-15    1234    953  764.782609       0.00
    857   2016-03-15    1461   1653  639.478261       0.00
    858   2016-03-15    1490    787  708.260870       0.25
    859   2016-03-15    1498    443  622.652174       0.00
    860   2016-03-15    1874    472  268.913043       0.00
    861   2016-03-15    2333   1107  634.000000       0.00
    862   2016-03-15    2346   2533  398.782609       0.00
    863   2016-03-15    2398   1234  912.608696       0.00
    864   2016-03-15    2667   3078  385.000000       1.50
    865   2016-03-15    2891  27124   18.869565       2.50
    866   2016-03-15    3308    674  854.347826       0.50
    868   2016-03-16     320   1639  432.000000       0.00
    869   2016-03-16     406   1088  981.000000       0.00
    870   2016-03-16     459   1643  816.000000       0.00
    871   2016-03-16     676   1036  822.000000       0.00
    872   2016-03-16     722   1288  778.000000       2.00
    873   2016-03-16    1234    929  757.000000       0.00
    874   2016-03-16    1461   1993  632.000000       0.50
    875   2016-03-16    1490    779  824.000000       0.00
    876   2016-03-16    1498    151  826.000000       0.25
    877   2016-03-16    1874     48  463.000000       0.00
    878   2016-03-16    2333    913  725.000000       0.00
    879   2016-03-16    2346   2490  429.000000       0.25
    880   2016-03-16    2398   2112  807.000000       0.25
    881   2016-03-16    2667   3049  402.000000       1.00
    882   2016-03-16    2891  30309   20.000000       2.50
    883   2016-03-16    3308    704  957.000000       0.00
    884   2016-03-16    3373  23134  121.000000       2.00
    885   2016-03-17     320   1826  471.521739       0.00
    886   2016-03-17     406   1266  882.695652       0.00
    887   2016-03-17     459   1674  632.869565       0.00
    888   2016-03-17     676   1286  758.347826       0.00
    889   2016-03-17     722   1518  778.347826       1.75
    890   2016-03-17    1234   1170  786.434783       0.25
    891   2016-03-17    1461   2531  593.826087       1.00
    892   2016-03-17    1490   1025  929.652174       2.00
    895   2016-03-17    2346   2808  477.695652       0.00
    896   2016-03-17    2398   1842  602.391304       0.00
    897   2016-03-17    2667   3495  426.434783       1.25
    898   2016-03-17    2891  34961   20.478261       7.75
    899   2016-03-17    3373  26072   70.565217       3.75
    900   2016-03-18     320   1680  499.434783       0.00
    901   2016-03-18     406   1114  894.434783       0.00
    902   2016-03-18     459   1160  569.391304       0.00
    903   2016-03-18     676   1154  719.695652       0.00
    904   2016-03-18     722   1367  788.956522       0.75
    905   2016-03-18    1234   1143  811.913043       0.00
    906   2016-03-18    1461   2260  539.130435       2.25
    908   2016-03-18    2346   2897  522.086957       0.50
    909   2016-03-18    2398   1793  636.826087       0.00
    910   2016-03-18    2667   3170  468.000000       1.25
    911   2016-03-18    2891  32913   23.652174       6.25
    912   2016-03-18    3373  20929   49.913043       4.50
    913   2016-03-19     320   1325  513.260870       0.00
    915   2016-03-19     459    698  705.956522       0.00
    916   2016-03-19     676    661  746.869565       0.00
    917   2016-03-19     722   1070  786.956522       0.00
    918   2016-03-19    1234    902  790.000000       0.25
    919   2016-03-19    1461   1566  546.304348       1.75
    920   2016-03-19    2346   2300  477.608696       0.50
    922   2016-03-19    2398   1238  618.739130       0.00
    923   2016-03-19    2667   3016  451.391304       0.75
    924   2016-03-19    2891  24902   23.217391       3.75
    926   2016-03-19    3373  15211   55.956522       3.00
    927   2016-03-20     320   1314  507.260870       0.00
    929   2016-03-20     459    985  802.913043       0.00
    930   2016-03-20     676    764  802.173913       0.00
    931   2016-03-20     722    997  814.565217       0.75
    932   2016-03-20    1234    840  805.304348       0.00
    933   2016-03-20    1461   1505  538.521739       0.75
    934   2016-03-20    2346   2374  474.173913       0.75
    935   2016-03-20    2373   1297  646.521739       0.50
    936   2016-03-20    2398   1352  680.956522       0.00
    937   2016-03-20    2667   2679  407.739130       0.00
    938   2016-03-20    2891  29736   23.217391       2.25
    940   2016-03-20    3373  15297   57.391304       3.50
    941   2016-03-21     320   1569  523.521739       0.25
    943   2016-03-21     459    786  747.565217       0.00
    944   2016-03-21     676    849  821.391304       0.25
    945   2016-03-21     722   2474  823.260870       0.50
    946   2016-03-21    1234    973  830.086957       0.00
    947   2016-03-21    1461   1522  606.086957       0.50
    950   2016-03-21    2346   2116  493.695652       0.50
    951   2016-03-21    2373   1069  708.565217       0.50
    952   2016-03-21    2398   1135  715.739130       0.00
    953   2016-03-21    2667   2648  422.913043       1.50
    954   2016-03-21    2891  29122   17.043478       4.25
    955   2016-03-21    3373  15809   59.565217       2.00
    956   2016-03-22     320   1392  487.869565       0.00
    957   2016-03-22     459   1920  850.130435       0.50
    958   2016-03-22     676    752  812.000000       0.00
    959   2016-03-22     722  12717  676.130435       1.00
    960   2016-03-22    1234    896  773.695652       0.00
    962   2016-03-22    1461   1192  621.913043       0.00
    963   2016-03-22    1490    726  883.347826       0.50
    964   2016-03-22    2333    892  867.826087       0.25
    965   2016-03-22    2346   2114  528.086957       0.25
    966   2016-03-22    2373    885  785.565217       1.25
    967   2016-03-22    2398    934  775.521739       0.25
    968   2016-03-22    2667   2740  472.521739       2.50
    969   2016-03-22    2891  27754   18.913043       3.75
    970   2016-03-22    3373   6649   60.086957       0.25
    972   2016-03-23     320   1514  486.130435       0.00
    973   2016-03-23     459   1469  561.260870       0.25
    974   2016-03-23     676    860  837.043478       0.00
    975   2016-03-23     722   7450  260.652174       0.75
    976   2016-03-23    1234    923  771.304348       0.25
    977   2016-03-23    1264    794  783.565217       0.25
    978   2016-03-23    1461   1103  700.869565       1.25
    979   2016-03-23    1490    940  903.347826       0.00
    981   2016-03-23    2346   2628  501.260870       0.00
    983   2016-03-23    2398   1021  873.391304       0.00
    984   2016-03-23    2667   2653  490.347826       0.25
    985   2016-03-23    2891  31568   19.000000       3.25
    986   2016-03-23    3373   5017  107.695652       1.25
    988   2016-03-24     320   1896  489.869565       0.25
    989   2016-03-24     459   1529  561.956522       0.50
    990   2016-03-24     676   1173  828.739130       0.00
    991   2016-03-24     722   5589  160.217391       0.25
    992   2016-03-24    1234   1154  803.000000       0.25
    994   2016-03-24    1461   1359  762.956522       1.50
    995   2016-03-24    1490   1121  861.652174       1.50
    996   2016-03-24    2346   2612  473.391304       0.25
    997   2016-03-24    2398   1253  888.695652       0.00
    998   2016-03-24    2667   3124  471.000000       0.25
    999   2016-03-24    2891  35771   20.521739       5.00
    1000  2016-03-24    3373   6328  181.391304       1.50
    1002  2016-03-25     320   1904  473.347826       0.00
    1003  2016-03-25     459   1733  611.086957       0.00
    1004  2016-03-25     676   1114  798.739130       0.00
    1005  2016-03-25     722   4362  229.260870       0.50
    1006  2016-03-25    1234   1125  812.782609       0.00
    1007  2016-03-25    1461   1204  797.608696       1.50
    1008  2016-03-25    1490   1009  896.173913       1.00
    1009  2016-03-25    2346   3617  485.086957       0.25
    1010  2016-03-25    2398   1266  919.304348       0.00
    1011  2016-03-25    2667   2958  496.478261       0.00
    1012  2016-03-25    2891  35380   21.826087       4.00
    1013  2016-03-25    3373   5478  211.782609       1.50
    1015  2016-03-26     320   1441  459.695652       0.00
    1016  2016-03-26     459   1508  559.478261       0.00
    1017  2016-03-26     676    708  794.565217       0.00
    1018  2016-03-26     722   2858  282.521739       0.75
    1019  2016-03-26    1234    840  800.434783       0.25
    1020  2016-03-26    1461    856  825.086957       0.00
    1021  2016-03-26    1490   2804  946.826087       0.50
    1022  2016-03-26    2346   2284  438.782609       0.25
    1023  2016-03-26    2398    859  881.695652       0.00
    1024  2016-03-26    2667   2744  503.478261       4.25
    1025  2016-03-26    2891  28448   20.869565       3.25
    1026  2016-03-26    3373   3938  234.173913       1.25
    1028  2016-03-27     320   1417  467.130435       0.25
    1029  2016-03-27     459   1104  515.521739       0.00
    1030  2016-03-27     676    740  846.000000       0.00
    1031  2016-03-27     722   2453  349.304348       1.75
    1032  2016-03-27    1234    853  800.478261       0.00
    1033  2016-03-27    1461    856  846.521739       0.50
    1034  2016-03-27    1490   3839  412.391304       0.50
    1035  2016-03-27    2346   2192  464.434783       0.50
    1036  2016-03-27    2398    778  906.130435       0.00
    1037  2016-03-27    2667   2786  487.391304       3.00
    1038  2016-03-27    2891  30854   19.956522       4.25
    1039  2016-03-27    3373   3834  248.913043       1.00
    1041  2016-03-28     320   1489  483.434783       0.25
    1042  2016-03-28     459    946  585.608696       0.00
    1043  2016-03-28     676    750  852.304348       0.00
    1044  2016-03-28     722   2510  406.826087       6.00
    1045  2016-03-28    1234    914  795.347826       0.00
    1046  2016-03-28    1461    847  891.434783       0.75
    1047  2016-03-28    1490   4023  242.913043       1.00
    1048  2016-03-28    2346   3930  424.260870       0.50
    1050  2016-03-28    2667   3276  451.608696       3.25
    1051  2016-03-28    2891  29605   17.652174       3.00
    1052  2016-03-28    3373   3952  247.347826       1.00
    1055  2016-03-29     320   1457  493.347826       0.00
    1056  2016-03-29     459   1060  668.608696       0.25
    1057  2016-03-29     676    963  856.000000       0.00
    1058  2016-03-29     722   1390  449.217391       2.50
    1059  2016-03-29    1234    998  788.086957       0.50
    1060  2016-03-29    1461    924  931.260870       1.25
    1061  2016-03-29    1490   1689  233.391304       1.25
    1063  2016-03-29    2346   3328  315.000000       0.50
    1065  2016-03-29    2667   3006  411.000000       1.00
    1066  2016-03-29    2891  30412   16.826087       3.25
    1067  2016-03-29    3373   3748  264.130435       1.75
    1070  2016-03-30     320   2236  494.391304       0.00
    1071  2016-03-30     459   1891  661.130435       0.00
    1072  2016-03-30     676   1497  736.043478       0.00
    1073  2016-03-30     722   1426  595.478261       1.75
    1074  2016-03-30    1234   2628  707.956522       0.00
    1076  2016-03-30    1461   1039  945.086957       0.50
    1077  2016-03-30    1490   1117  431.086957       1.75
    1078  2016-03-30    1874   1364  550.826087       0.00
    1079  2016-03-30    2346   2926  337.086957       0.25
    1081  2016-03-30    2398   1297  917.565217       0.00
    1082  2016-03-30    2667   3257  455.391304       1.75
    1085  2016-03-30    3373   3859  269.000000       1.00
    1088  2016-03-31     320   2041  445.000000       0.00
    1089  2016-03-31     459   1526  506.130435       0.00
    1090  2016-03-31     676   1157  622.652174       0.00
    1091  2016-03-31     722   1388  673.521739       0.50
    1092  2016-03-31    1234   1800  500.913043       0.00
    1093  2016-03-31    1264   2433  954.826087       0.75
    1094  2016-03-31    1461   1022  905.434783       0.50
    1095  2016-03-31    1490   1223  625.434783       0.50
    1096  2016-03-31    1874    237  579.869565       0.00
    1097  2016-03-31    2346  11099  389.565217       2.00
    1098  2016-03-31    2398   1372  918.217391       0.00
    1099  2016-03-31    2667   3294  479.130435       1.75
    1100  2016-03-31    2891  35903   21.739130       7.75
    1101  2016-03-31    3308    777  944.652174       0.25
    1102  2016-03-31    3373   3770  312.956522       3.25
    


```python
newData.avgrating.dtype
```




    dtype('float64')



Further we noticed that avg rating column has a lot of values set as 0.00 . We can always replace these null values by a normalize function or we can also replace them by mean. But, replcaing them by either of them in this case, will bias the ML model. We want our model to be trained in such a way, that it can interpret that low ranks have higher sales. For the ease of use, we will only consider avg rank as a factor influencing the sales.


```python
newData1 = newData[['Date','sales','avgrank']]
print(newData1)
```

                Date  sales     avgrank
    0     2016-01-01   2412  469.739130
    1     2016-01-01   1308  719.173913
    2     2016-01-01   2037  603.695652
    3     2016-01-01   2052  567.913043
    4     2016-01-01   1553  776.826087
    5     2016-01-01   2152  518.260870
    7     2016-01-01  89289    5.782609
    8     2016-01-02   1622  464.043478
    9     2016-01-02   1110  754.652174
    10    2016-01-02   1553  586.217391
    11    2016-01-02   1467  617.782609
    12    2016-01-02   1011  727.695652
    13    2016-01-02   1722  546.260870
    16    2016-01-02  84915    5.826087
    17    2016-01-03   1564  508.695652
    18    2016-01-03    963  762.434783
    19    2016-01-03   1409  560.521739
    20    2016-01-03   1371  676.000000
    21    2016-01-03    938  829.217391
    22    2016-01-03   1662  560.913043
    23    2016-01-03   2029  680.086957
    24    2016-01-03  61345    5.000000
    25    2016-01-04   1609  525.695652
    26    2016-01-04   1168  797.217391
    27    2016-01-04   1732  573.826087
    28    2016-01-04   1421  715.043478
    29    2016-01-04    989  840.652174
    30    2016-01-04   1081  573.782609
    31    2016-01-04   1460  654.565217
    32    2016-01-04  58943    6.391304
    33    2016-01-05   1594  548.173913
    34    2016-01-05   1163  750.869565
    35    2016-01-05   1392  563.130435
    36    2016-01-05   1482  757.086957
    37    2016-01-05   1055  875.173913
    38    2016-01-05    984  743.217391
    39    2016-01-05   1102  807.869565
    40    2016-01-05  62196   10.782609
    41    2016-01-06   1883  545.739130
    42    2016-01-06   1252  742.173913
    44    2016-01-06   1737  706.608696
    45    2016-01-06   1227  855.782609
    46    2016-01-06   1244  848.260870
    49    2016-01-06  67759   10.478261
    50    2016-01-07   1978  527.478261
    51    2016-01-07   1254  763.217391
    53    2016-01-07   1538  701.173913
    54    2016-01-07   1125  845.086957
    55    2016-01-07   1184  878.869565
    56    2016-01-07    956  940.260870
    57    2016-01-07  60462    9.434783
    58    2016-01-08   1509  483.608696
    59    2016-01-08    960  736.043478
    61    2016-01-08   1273  702.695652
    62    2016-01-08    893  818.695652
    63    2016-01-08    876  899.391304
    65    2016-01-08  36569   11.608696
    66    2016-01-09   1256  502.652174
    67    2016-01-09    761  783.478261
    68    2016-01-09    867  657.913043
    69    2016-01-09   1038  705.956522
    70    2016-01-09    808  789.695652
    71    2016-01-09    871  872.608696
    72    2016-01-09  28872   16.695652
    73    2016-01-10   1316  519.652174
    74    2016-01-10    775  815.913043
    76    2016-01-10   1248  702.260870
    78    2016-01-10    795  804.869565
    79    2016-01-10    866  844.000000
    80    2016-01-10  32229   20.608696
    81    2016-01-11   1350  524.173913
    82    2016-01-11    744  832.000000
    83    2016-01-11   1454  430.869565
    84    2016-01-11   1387  651.565217
    85    2016-01-11   1008  857.695652
    86    2016-01-11    879  824.782609
    87    2016-01-11   1505  811.608696
    90    2016-01-11  30459   19.086957
    91    2016-01-12   1650  525.478261
    92    2016-01-12   1071  867.130435
    93    2016-01-12   1589  481.826087
    94    2016-01-12   1622  588.739130
    95    2016-01-12    909  923.565217
    96    2016-01-12    946  821.347826
    97    2016-01-12   2339  675.521739
    98    2016-01-12   2047  774.913043
    100   2016-01-12  30254   21.608696
    122   2016-01-14   2046  562.434783
    123   2016-01-14   1285  824.391304
    124   2016-01-14   1893  480.826087
    129   2016-01-14   2371  636.304348
    134   2016-01-14  49142   19.782609
    135   2016-01-15   1813  531.565217
    136   2016-01-15   1132  843.869565
    137   2016-01-15   1478  556.695652
    138   2016-01-15   1613  686.086957
    139   2016-01-15   1110  810.347826
    140   2016-01-15   2081  637.739130
    141   2016-01-15   1769  728.521739
    142   2016-01-15  37157   19.173913
    143   2016-01-16   1372  493.043478
    144   2016-01-16    844  834.304348
    146   2016-01-16   1391  685.695652
    147   2016-01-16    900  826.260870
    148   2016-01-16   1554  639.260870
    149   2016-01-16   1443  757.043478
    150   2016-01-16  30890   23.956522
    151   2016-01-17   1385  519.478261
    152   2016-01-17    892  823.521739
    153   2016-01-17    977  624.521739
    154   2016-01-17   2729  606.086957
    155   2016-01-17    919  828.739130
    156   2016-01-17   1673  661.000000
    157   2016-01-17   1115  787.478261
    159   2016-01-17  27467   25.695652
    160   2016-01-18   1291  546.608696
    161   2016-01-18    777  841.695652
    163   2016-01-18   3084  433.347826
    164   2016-01-18    804  809.391304
    165   2016-01-18    943  678.217391
    167   2016-01-18  44326   68.956522
    168   2016-01-18  22792   28.304348
    169   2016-01-19   1368  545.565217
    170   2016-01-19    945  879.913043
    171   2016-01-19   1977  733.652174
    172   2016-01-19   2754  383.173913
    173   2016-01-19    920  835.869565
    174   2016-01-19   1094  788.434783
    176   2016-01-19  47512   25.739130
    177   2016-01-19  29160   32.608696
    178   2016-01-20   1746  565.173913
    179   2016-01-20   1267  876.347826
    181   2016-01-20   3107  390.652174
    182   2016-01-20   1096  859.695652
    183   2016-01-20   1319  749.478261
    184   2016-01-20   1355  766.130435
    186   2016-01-20  49850   24.826087
    187   2016-01-20  36121   33.000000
    188   2016-01-21   1828  555.695652
    189   2016-01-21   1124  805.956522
    190   2016-01-21   1923  401.782609
    191   2016-01-21   2897  407.304348
    192   2016-01-21   1220  848.739130
    193   2016-01-21   1399  714.043478
    194   2016-01-21   1582  854.217391
    195   2016-01-21   1958  838.695652
    196   2016-01-21  45371   27.652174
    197   2016-01-21  34248   35.478261
    198   2016-01-22   1232  514.913043
    199   2016-01-22    727  821.478261
    200   2016-01-22   1532  462.260870
    201   2016-01-22   2090  431.043478
    202   2016-01-22    842  768.130435
    203   2016-01-22   1043  692.086957
    204   2016-01-22   1284  892.043478
    205   2016-01-22   1654  667.043478
    206   2016-01-22  28863   29.000000
    207   2016-01-22  21218   35.913043
    208   2016-01-23   1167  502.173913
    209   2016-01-23    798  847.739130
    210   2016-01-23   1198  479.521739
    211   2016-01-23   2135  433.043478
    212   2016-01-23    775  769.173913
    213   2016-01-23    856  679.826087
    214   2016-01-23   1276  794.826087
    215   2016-01-23   1192  629.391304
    218   2016-01-24   1189  532.695652
    219   2016-01-24    708  862.695652
    220   2016-01-24   2028  474.652174
    221   2016-01-24   1781  450.130435
    223   2016-01-24    794  802.521739
    224   2016-01-24    886  737.347826
    225   2016-01-24   1120  750.782609
    226   2016-01-24   1168  645.434783
    227   2016-01-24  21527   36.782609
    228   2016-01-24  18949   32.826087
    229   2016-01-25   1250  539.304348
    230   2016-01-25    779  894.130435
    231   2016-01-25   1336  423.304348
    232   2016-01-25   1703  501.217391
    233   2016-01-25   1019  928.956522
    234   2016-01-25    791  870.000000
    235   2016-01-25    935  732.086957
    236   2016-01-25   1087  811.565217
    237   2016-01-25   1259  726.565217
    238   2016-01-25  16278   46.739130
    239   2016-01-25  20721   36.869565
    240   2016-01-26   1368  545.782609
    241   2016-01-26    924  902.130435
    242   2016-01-26   1431  520.347826
    243   2016-01-26   2316  507.782609
    244   2016-01-26    949  932.826087
    245   2016-01-26    931  843.956522
    246   2016-01-26   1029  723.521739
    248   2016-01-26   1204  702.347826
    249   2016-01-26  16522   63.478261
    250   2016-01-26  27454   33.217391
    251   2016-01-27   1807  553.434783
    252   2016-01-27   1317  874.086957
    253   2016-01-27   2079  520.347826
    254   2016-01-27   2487  482.565217
    256   2016-01-27   1157  853.130435
    257   2016-01-27   1213  757.086957
    259   2016-01-27   1758  746.652174
    260   2016-01-27  18981   77.739130
    261   2016-01-27  34159   31.391304
    262   2016-01-28   1639  554.608696
    263   2016-01-28   1188  799.391304
    264   2016-01-28   2361  514.521739
    265   2016-01-28   2116  505.304348
    266   2016-01-28   1147  840.000000
    267   2016-01-28   1321  781.956522
    269   2016-01-28   1935  726.347826
    270   2016-01-28  17933   85.260870
    271   2016-01-28  34537   30.826087
    272   2016-01-29   1240  537.173913
    273   2016-01-29    703  816.304348
    274   2016-01-29   1480  456.391304
    275   2016-01-29   1502  540.739130
    276   2016-01-29    827  842.521739
    277   2016-01-29    863  748.565217
    278   2016-01-29   7055  164.739130
    279   2016-01-29    386  736.652174
    280   2016-01-29  13235   88.956522
    281   2016-01-29  20230   30.913043
    282   2016-01-30   1175  529.652174
    283   2016-01-30    667  868.956522
    284   2016-01-30   1409  467.521739
    285   2016-01-30   1727  564.826087
    286   2016-01-30    797  824.173913
    287   2016-01-30   1197  789.565217
    289   2016-01-30   7829  138.565217
    291   2016-01-30  13021   87.304348
    292   2016-01-30  16766   38.478261
    294   2016-01-31   1149  532.739130
    295   2016-01-31    711  896.478261
    296   2016-01-31   1207  495.521739
    297   2016-01-31   2059  475.086957
    298   2016-01-31    790  781.086957
    299   2016-01-31    739  676.043478
    301   2016-01-31   6435  128.565217
    302   2016-01-31  11656   91.043478
    303   2016-01-31  15648   45.173913
    304   2016-02-01   1270  547.086957
    305   2016-02-01    773  884.434783
    306   2016-02-01   1271  525.869565
    307   2016-02-01   1649  473.521739
    309   2016-02-01    928  777.652174
    310   2016-02-01    751  772.565217
    312   2016-02-01   6630  134.260870
    313   2016-02-01   8485  105.260870
    314   2016-02-01  16267   50.565217
    316   2016-02-02   1435  549.434783
    317   2016-02-02   1003  873.434783
    318   2016-02-02   1181  588.391304
    319   2016-02-02   1770  505.000000
    320   2016-02-02   1080  747.652174
    323   2016-02-02   4731  155.739130
    324   2016-02-02   8677  145.565217
    325   2016-02-02  18581   56.521739
    327   2016-02-03   1858  563.478261
    328   2016-02-03   1419  817.521739
    329   2016-02-03   1409  630.000000
    330   2016-02-03   2330  540.521739
    331   2016-02-03   1361  768.043478
    333   2016-02-03   2569  765.043478
    334   2016-02-03   6081  197.130435
    335   2016-02-03   9948  163.565217
    336   2016-02-03  25123   57.173913
    339   2016-02-04   1869  537.782609
    340   2016-02-04   1312  813.956522
    341   2016-02-04   1678  648.086957
    342   2016-02-04   2164  550.304348
    343   2016-02-04   1254  737.217391
    345   2016-02-04   3304  621.826087
    346   2016-02-04   4468  228.086957
    347   2016-02-04   9212  183.956522
    348   2016-02-04  23394   59.565217
    349   2016-02-04   6235  262.086957
    351   2016-02-05   1320  520.086957
    352   2016-02-05    846  800.695652
    353   2016-02-05   1293  616.869565
    354   2016-02-05   1449  561.782609
    355   2016-02-05    925  754.173913
    356   2016-02-05   2616  541.304348
    357   2016-02-05   2302  293.695652
    358   2016-02-05   6666  186.130435
    359   2016-02-05  14004   60.043478
    360   2016-02-05   3760  248.391304
    362   2016-02-06   1280  533.695652
    363   2016-02-06    769  808.304348
    364   2016-02-06   1014  612.695652
    365   2016-02-06   1769  557.695652
    366   2016-02-06    925  759.521739
    367   2016-02-06   2343  530.391304
    368   2016-02-06   1973  386.956522
    369   2016-02-06   6154  177.565217
    370   2016-02-06  13272   65.478261
    371   2016-02-06   3408  295.260870
    372   2016-02-06   3552  364.260870
    373   2016-02-07   1439  536.434783
    374   2016-02-07    929  850.434783
    375   2016-02-07    979  654.217391
    376   2016-02-07   1663  515.086957
    377   2016-02-07    989  789.565217
    378   2016-02-07   2189  521.739130
    379   2016-02-07   1938  458.608696
    380   2016-02-07   6039  181.217391
    381   2016-02-07  12973   69.869565
    382   2016-02-07   3955  334.391304
    383   2016-02-07   3062  379.000000
    384   2016-02-08   1359  530.869565
    385   2016-02-08    815  838.304348
    386   2016-02-08   1439  692.521739
    387   2016-02-08   1667  554.956522
    388   2016-02-08    927  768.913043
    389   2016-02-08   2127  516.391304
    390   2016-02-08   1618  515.260870
    391   2016-02-08   5211  193.304348
    392   2016-02-08  11992   78.173913
    393   2016-02-08   1317  335.000000
    394   2016-02-08   2002  435.304348
    395   2016-02-09   1735  521.478261
    396   2016-02-09   1020  816.739130
    397   2016-02-09   1465  580.739130
    398   2016-02-09   2006  607.391304
    400   2016-02-09   1088  752.304348
    401   2016-02-09   2027  548.434783
    402   2016-02-09   1568  558.826087
    403   2016-02-09   5879  210.043478
    404   2016-02-09  15179   78.130435
    405   2016-02-09    556  606.913043
    406   2016-02-09   2652  537.695652
    407   2016-02-10   2212  527.739130
    408   2016-02-10   1385  764.434783
    409   2016-02-10   1697  581.565217
    410   2016-02-10   2673  575.869565
    411   2016-02-10   1448  762.478261
    412   2016-02-10   3024  627.956522
    413   2016-02-10   1736  650.608696
    414   2016-02-10   6547  229.260870
    415   2016-02-10  20504   80.869565
    418   2016-02-10   2061  569.391304
    419   2016-02-11   2064  481.347826
    420   2016-02-11   1238  785.565217
    421   2016-02-11   1760  622.608696
    422   2016-02-11   2236  552.000000
    423   2016-02-11   1381  714.173913
    424   2016-02-11   3513  602.695652
    425   2016-02-11   1719  707.347826
    426   2016-02-11   6110  261.956522
    427   2016-02-11  17485   76.695652
    429   2016-02-11   1916  794.652174
    430   2016-02-12   1449  467.000000
    431   2016-02-12    791  825.173913
    432   2016-02-12   1235  604.130435
    433   2016-02-12   1489  593.043478
    434   2016-02-12   1001  709.000000
    435   2016-02-12   2534  523.347826
    436   2016-02-12   1232  723.000000
    437   2016-02-12   4729  272.521739
    438   2016-02-12  11147   79.086957
    442   2016-02-13   1342  487.478261
    443   2016-02-13    702  858.956522
    444   2016-02-13   1728  563.434783
    445   2016-02-13   1380  606.695652
    447   2016-02-13   1009  678.956522
    448   2016-02-13   2332  462.000000
    450   2016-02-13   1271  739.304348
    451   2016-02-13   4395  261.391304
    452   2016-02-13   9598   82.260870
    454   2016-02-14   2928  223.304348
    455   2016-02-14   1280  506.347826
    456   2016-02-14    674  904.260870
    457   2016-02-14    949  473.565217
    458   2016-02-14   1353  625.956522
    459   2016-02-14    746  924.086957
    460   2016-02-14    877  684.695652
    461   2016-02-14   1954  451.130435
    462   2016-02-14    826  742.739130
    463   2016-02-14   1090  701.347826
    464   2016-02-14   4079  261.043478
    465   2016-02-14  10271   87.956522
    468   2016-02-15   2029  278.826087
    469   2016-02-15   1430  485.391304
    470   2016-02-15    768  902.086957
    471   2016-02-15    940  562.173913
    472   2016-02-15   1441  606.608696
    474   2016-02-15    969  692.000000
    476   2016-02-15   2134  479.173913
    478   2016-02-15   1316  726.478261
    479   2016-02-15   4924  253.956522
    480   2016-02-15  11795   78.000000
    483   2016-02-16   1851  415.913043
    484   2016-02-16   1582  480.956522
    485   2016-02-16   1000  888.434783
    486   2016-02-16   1409  644.608696
    487   2016-02-16   1997  598.000000
    488   2016-02-16   1079  726.913043
    489   2016-02-16   1562  867.652174
    490   2016-02-16   2062  503.173913
    491   2016-02-16   1478  710.826087
    492   2016-02-16   5353  251.782609
    493   2016-02-16  15537   75.478261
    494   2016-02-17   1723  561.956522
    495   2016-02-17   1919  503.304348
    496   2016-02-17   1307  865.000000
    497   2016-02-17   1661  582.913043
    498   2016-02-17   2217  562.391304
    499   2016-02-17   1327  750.565217
    500   2016-02-17   1960  807.521739
    501   2016-02-17   2706  569.739130
    502   2016-02-17   1865  747.043478
    503   2016-02-17   8867  265.608696
    504   2016-02-17  20417   74.652174
    506   2016-02-18   1419  705.086957
    507   2016-02-18   2025  506.782609
    508   2016-02-18   1293  802.217391
    509   2016-02-18   1821  584.217391
    510   2016-02-18   2385  574.869565
    511   2016-02-18   1310  771.304348
    512   2016-02-18   1734  835.739130
    513   2016-02-18   3556  552.391304
    514   2016-02-18   2286  733.695652
    515   2016-02-18  35279  184.565217
    516   2016-02-18  20284   72.043478
    518   2016-02-19   1357  798.869565
    519   2016-02-19   1560  505.173913
    520   2016-02-19   1021  810.000000
    521   2016-02-19    715  628.000000
    522   2016-02-19   1969  533.086957
    523   2016-02-19   1170  775.478261
    524   2016-02-19   1476  859.521739
    525   2016-02-19   3099  496.043478
    526   2016-02-19   1933  668.304348
    527   2016-02-19  16330   78.826087
    528   2016-02-19  19032   71.130435
    531   2016-02-20   1302  513.260870
    532   2016-02-20    856  844.695652
    533   2016-02-20   1271  794.260870
    534   2016-02-20   1555  542.043478
    535   2016-02-20    946  780.913043
    536   2016-02-20   1163  839.000000
    537   2016-02-20   3295  443.869565
    538   2016-02-20   1310  670.000000
    539   2016-02-20  10265   98.652174
    540   2016-02-20  14375   64.173913
    542   2016-02-21   1357  508.869565
    543   2016-02-21    799  826.521739
    544   2016-02-21   1187  679.913043
    545   2016-02-21   1413  560.130435
    546   2016-02-21    972  771.695652
    547   2016-02-21   1138  880.826087
    548   2016-02-21   3777  381.652174
    550   2016-02-21   1256  706.695652
    551   2016-02-21   8114  119.260870
    552   2016-02-21  14902   63.478261
    554   2016-02-22   1396  508.086957
    555   2016-02-22    789  821.217391
    556   2016-02-22   1086  623.347826
    557   2016-02-22   1486  600.391304
    558   2016-02-22   1024  735.217391
    559   2016-02-22    806  882.956522
    560   2016-02-22   5447  321.782609
    561   2016-02-22    820  657.739130
    562   2016-02-22   1297  741.391304
    563   2016-02-22   7352  150.478261
    564   2016-02-22  18591   60.608696
    567   2016-02-23   1654  499.739130
    568   2016-02-23   1059  848.130435
    569   2016-02-23   1381  589.217391
    570   2016-02-23   1763  610.130435
    572   2016-02-23   1149  717.173913
    574   2016-02-23   3896  349.652174
    576   2016-02-23   1469  736.043478
    577   2016-02-23   7706  163.869565
    578   2016-02-23  21589   45.000000
    581   2016-02-24   2128  492.913043
    582   2016-02-24   1373  777.217391
    583   2016-02-24   1614  581.608696
    585   2016-02-24   2165  589.608696
    586   2016-02-24   1373  706.869565
    588   2016-02-24   3034  431.347826
    589   2016-02-24   2178  718.782609
    590   2016-02-24   8076  187.347826
    592   2016-02-24  29501   40.173913
    593   2016-02-25   1936  477.739130
    594   2016-02-25   1227  760.391304
    595   2016-02-25   1717  638.000000
    596   2016-02-25    983  933.434783
    597   2016-02-25   1981  595.086957
    598   2016-02-25   1319  734.652174
    600   2016-02-25   3048  529.739130
    601   2016-02-25   1689  693.478261
    602   2016-02-25   5546  221.956522
    603   2016-02-25   1647  619.304348
    604   2016-02-25  31090   35.304348
    605   2016-02-26   1480  468.173913
    606   2016-02-26    811  774.130435
    607   2016-02-26   1055  620.739130
    608   2016-02-26    627  925.173913
    609   2016-02-26   1183  612.086957
    610   2016-02-26    962  725.130435
    612   2016-02-26   2712  509.913043
    613   2016-02-26   1020  762.695652
    614   2016-02-26   4138  266.304348
    615   2016-02-26    176  648.521739
    616   2016-02-26  16154   34.521739
    618   2016-02-27   1458  422.956522
    619   2016-02-27    700  785.695652
    620   2016-02-27    865  646.869565
    621   2016-02-27    563  934.956522
    622   2016-02-27   1314  668.826087
    624   2016-02-27    892  696.913043
    625   2016-02-27   2770  426.565217
    626   2016-02-27   1199  781.130435
    627   2016-02-27   4485  271.695652
    629   2016-02-27  15148   41.130435
    632   2016-02-28   1430  414.217391
    633   2016-02-28    768  803.000000
    634   2016-02-28   1364  670.347826
    636   2016-02-28   1411  618.869565
    638   2016-02-28    951  708.826087
    639   2016-02-28   2600  374.608696
    640   2016-02-28   1438  758.260870
    641   2016-02-28   4460  260.695652
    642   2016-02-28  15738   43.478261
    643   2016-02-28   1047  880.260870
    645   2016-03-01   1586  423.913043
    646   2016-03-01    853  796.217391
    647   2016-03-01   1590  560.869565
    648   2016-03-01    662  957.086957
    649   2016-03-01   3075  571.608696
    651   2016-03-01    979  683.739130
    653   2016-03-01   1548  398.913043
    654   2016-03-01   1401  706.304348
    655   2016-03-01   5233  245.565217
    656   2016-03-01  19151   39.782609
    657   2016-03-01   1200  745.434783
    659   2016-03-02   2041  417.260870
    660   2016-03-02   1012  813.565217
    661   2016-03-02   1786  506.043478
    662   2016-03-02    823  928.217391
    663   2016-03-02   4436  431.434783
    664   2016-03-02   1114  681.217391
    667   2016-03-02   4691  477.260870
    668   2016-03-02   1542  537.043478
    669   2016-03-02   2221  635.086957
    670   2016-03-02   5296  223.304348
    671   2016-03-02  25378   34.826087
    672   2016-03-02   1252  663.565217
    673   2016-03-03   2322  393.304348
    674   2016-03-03   1277  805.869565
    675   2016-03-03   1508  524.391304
    676   2016-03-03   1049  924.826087
    677   2016-03-03   5262  371.695652
    678   2016-03-03   1294  716.826087
    679   2016-03-03   1317  872.565217
    680   2016-03-03   6659  327.434783
    681   2016-03-03   1706  683.217391
    682   2016-03-03   2198  505.304348
    683   2016-03-03   5575  253.652174
    684   2016-03-03  33408   29.565217
    685   2016-03-03   1982  662.434783
    687   2016-03-04   2084  404.826087
    688   2016-03-04   1123  801.304348
    689   2016-03-04   1720  594.521739
    690   2016-03-04    990  915.913043
    691   2016-03-04   4766  364.130435
    692   2016-03-04   1396  752.217391
    693   2016-03-04   1115  904.434783
    694   2016-03-04   6417  253.739130
    695   2016-03-04   1925  776.739130
    696   2016-03-04   1691  573.000000
    698   2016-03-04   5401  291.782609
    699   2016-03-04  32011   30.565217
    700   2016-03-04   2568  612.521739
    701   2016-03-05   1581  405.043478
    702   2016-03-05    664  823.173913
    703   2016-03-05   1888  566.000000
    704   2016-03-05    631  892.000000
    705   2016-03-05   3108  368.608696
    707   2016-03-05   3348  572.695652
    708   2016-03-05   1131  890.869565
    709   2016-03-05   4743  255.695652
    710   2016-03-05   1862  716.565217
    711   2016-03-05    937  695.913043
    713   2016-03-05   3862  293.869565
    714   2016-03-05  32532   28.869565
    715   2016-03-05   2426  496.521739
    717   2016-03-06   1527  410.434783
    718   2016-03-06    757  886.260870
    719   2016-03-06   1526  437.260870
    720   2016-03-06    786  832.913043
    721   2016-03-06   3066  382.000000
    722   2016-03-06    928  922.652174
    723   2016-03-06   1710  350.260870
    724   2016-03-06   1391  779.956522
    726   2016-03-06   2689  246.260870
    727   2016-03-06   1827  603.695652
    728   2016-03-06    990  778.913043
    729   2016-03-06   4908  289.695652
    730   2016-03-06  33458   17.130435
    731   2016-03-06   1779  412.652174
    733   2016-03-07   1603  431.521739
    734   2016-03-07    798  928.130435
    735   2016-03-07   1207  496.043478
    736   2016-03-07   1008  780.260870
    737   2016-03-07   3151  398.304348
    739   2016-03-07   1368  425.695652
    740   2016-03-07   1529  659.130435
    741   2016-03-07   1731  354.260870
    742   2016-03-07   2077  595.000000
    743   2016-03-07   1188  812.000000
    744   2016-03-07   4436  257.869565
    745   2016-03-07  34919   14.521739
    746   2016-03-07   1640  447.478261
    747   2016-03-08   1469  422.608696
    748   2016-03-08    711  898.434783
    749   2016-03-08   1096  612.956522
    750   2016-03-08    757  702.347826
    751   2016-03-08   2009  415.086957
    752   2016-03-08   1082  553.043478
    753   2016-03-08   1259  593.217391
    754   2016-03-08   2015  562.565217
    755   2016-03-08   2057  536.130435
    756   2016-03-08   1143  768.130435
    757   2016-03-08   3709  267.260870
    758   2016-03-08  33293   13.652174
    759   2016-03-08   1551  504.565217
    760   2016-03-09   1639  436.086957
    761   2016-03-09    957  930.086957
    762   2016-03-09   1567  623.260870
    763   2016-03-09    859  779.782609
    764   2016-03-09   1851  507.869565
    765   2016-03-09   1259  628.869565
    767   2016-03-09   1520  613.260870
    769   2016-03-09   2084  595.000000
    770   2016-03-09   1915  497.652174
    771   2016-03-09   1290  829.478261
    772   2016-03-09   3816  303.826087
    773   2016-03-09  35441   13.565217
    774   2016-03-09   2183  473.173913
    775   2016-03-10   1919  475.391304
    776   2016-03-10   1280  902.000000
    777   2016-03-10   1869  532.521739
    778   2016-03-10   1209  801.130435
    779   2016-03-10   2006  518.043478
    780   2016-03-10   1477  645.000000
    781   2016-03-10    998  877.217391
    782   2016-03-10   1766  607.434783
    783   2016-03-10  19010  135.608696
    784   2016-03-10   2478  618.000000
    785   2016-03-10   2616  571.478261
    786   2016-03-10   1957  774.956522
    787   2016-03-10   4539  333.391304
    788   2016-03-10  41386   15.826087
    789   2016-03-10   2946  408.347826
    805   2016-03-12   1395  481.565217
    806   2016-03-12    895  906.260870
    807   2016-03-12   1154  575.956522
    808   2016-03-12    919  752.565217
    809   2016-03-12   1231  650.043478
    810   2016-03-12    874  730.173913
    812   2016-03-12   1602  657.086957
    813   2016-03-12  16656   45.130435
    814   2016-03-12   1771  557.260870
    815   2016-03-12   2661  457.869565
    816   2016-03-12    993  873.608696
    817   2016-03-12   3257  365.086957
    818   2016-03-12  28069   23.000000
    819   2016-03-12   1447  404.913043
    820   2016-03-13   1772  477.130435
    821   2016-03-13    833  891.086957
    822   2016-03-13   1127  615.434783
    823   2016-03-13   1036  747.739130
    824   2016-03-13   1188  691.826087
    825   2016-03-13    898  764.565217
    826   2016-03-13   2082  442.956522
    827   2016-03-13   1502  599.130435
    828   2016-03-13   4107   52.217391
    829   2016-03-13   1385  595.260870
    830   2016-03-13   2750  439.130435
    831   2016-03-13   1027  931.608696
    832   2016-03-13   2941  371.608696
    833   2016-03-13  28991   22.826087
    834   2016-03-13    872  531.130435
    835   2016-03-14   1528  431.565217
    836   2016-03-14    738  921.869565
    837   2016-03-14   1084  654.652174
    838   2016-03-14    871  783.043478
    839   2016-03-14   1163  734.521739
    840   2016-03-14    880  771.608696
    841   2016-03-14   1433  507.608696
    842   2016-03-14   1039  607.478261
    844   2016-03-14   1761  116.565217
    845   2016-03-14   1777  683.217391
    846   2016-03-14   2917  411.739130
    847   2016-03-14    909  891.173913
    848   2016-03-14   3078  405.478261
    849   2016-03-14  27801   20.782609
    850   2016-03-14    815  710.913043
    851   2016-03-15   1614  416.173913
    852   2016-03-15    758  931.260870
    853   2016-03-15    873  689.608696
    854   2016-03-15    743  778.521739
    855   2016-03-15   1139  751.695652
    856   2016-03-15    953  764.782609
    857   2016-03-15   1653  639.478261
    858   2016-03-15    787  708.260870
    859   2016-03-15    443  622.652174
    860   2016-03-15    472  268.913043
    861   2016-03-15   1107  634.000000
    862   2016-03-15   2533  398.782609
    863   2016-03-15   1234  912.608696
    864   2016-03-15   3078  385.000000
    865   2016-03-15  27124   18.869565
    866   2016-03-15    674  854.347826
    868   2016-03-16   1639  432.000000
    869   2016-03-16   1088  981.000000
    870   2016-03-16   1643  816.000000
    871   2016-03-16   1036  822.000000
    872   2016-03-16   1288  778.000000
    873   2016-03-16    929  757.000000
    874   2016-03-16   1993  632.000000
    875   2016-03-16    779  824.000000
    876   2016-03-16    151  826.000000
    877   2016-03-16     48  463.000000
    878   2016-03-16    913  725.000000
    879   2016-03-16   2490  429.000000
    880   2016-03-16   2112  807.000000
    881   2016-03-16   3049  402.000000
    882   2016-03-16  30309   20.000000
    883   2016-03-16    704  957.000000
    884   2016-03-16  23134  121.000000
    885   2016-03-17   1826  471.521739
    886   2016-03-17   1266  882.695652
    887   2016-03-17   1674  632.869565
    888   2016-03-17   1286  758.347826
    889   2016-03-17   1518  778.347826
    890   2016-03-17   1170  786.434783
    891   2016-03-17   2531  593.826087
    892   2016-03-17   1025  929.652174
    895   2016-03-17   2808  477.695652
    896   2016-03-17   1842  602.391304
    897   2016-03-17   3495  426.434783
    898   2016-03-17  34961   20.478261
    899   2016-03-17  26072   70.565217
    900   2016-03-18   1680  499.434783
    901   2016-03-18   1114  894.434783
    902   2016-03-18   1160  569.391304
    903   2016-03-18   1154  719.695652
    904   2016-03-18   1367  788.956522
    905   2016-03-18   1143  811.913043
    906   2016-03-18   2260  539.130435
    908   2016-03-18   2897  522.086957
    909   2016-03-18   1793  636.826087
    910   2016-03-18   3170  468.000000
    911   2016-03-18  32913   23.652174
    912   2016-03-18  20929   49.913043
    913   2016-03-19   1325  513.260870
    915   2016-03-19    698  705.956522
    916   2016-03-19    661  746.869565
    917   2016-03-19   1070  786.956522
    918   2016-03-19    902  790.000000
    919   2016-03-19   1566  546.304348
    920   2016-03-19   2300  477.608696
    922   2016-03-19   1238  618.739130
    923   2016-03-19   3016  451.391304
    924   2016-03-19  24902   23.217391
    926   2016-03-19  15211   55.956522
    927   2016-03-20   1314  507.260870
    929   2016-03-20    985  802.913043
    930   2016-03-20    764  802.173913
    931   2016-03-20    997  814.565217
    932   2016-03-20    840  805.304348
    933   2016-03-20   1505  538.521739
    934   2016-03-20   2374  474.173913
    935   2016-03-20   1297  646.521739
    936   2016-03-20   1352  680.956522
    937   2016-03-20   2679  407.739130
    938   2016-03-20  29736   23.217391
    940   2016-03-20  15297   57.391304
    941   2016-03-21   1569  523.521739
    943   2016-03-21    786  747.565217
    944   2016-03-21    849  821.391304
    945   2016-03-21   2474  823.260870
    946   2016-03-21    973  830.086957
    947   2016-03-21   1522  606.086957
    950   2016-03-21   2116  493.695652
    951   2016-03-21   1069  708.565217
    952   2016-03-21   1135  715.739130
    953   2016-03-21   2648  422.913043
    954   2016-03-21  29122   17.043478
    955   2016-03-21  15809   59.565217
    956   2016-03-22   1392  487.869565
    957   2016-03-22   1920  850.130435
    958   2016-03-22    752  812.000000
    959   2016-03-22  12717  676.130435
    960   2016-03-22    896  773.695652
    962   2016-03-22   1192  621.913043
    963   2016-03-22    726  883.347826
    964   2016-03-22    892  867.826087
    965   2016-03-22   2114  528.086957
    966   2016-03-22    885  785.565217
    967   2016-03-22    934  775.521739
    968   2016-03-22   2740  472.521739
    969   2016-03-22  27754   18.913043
    970   2016-03-22   6649   60.086957
    972   2016-03-23   1514  486.130435
    973   2016-03-23   1469  561.260870
    974   2016-03-23    860  837.043478
    975   2016-03-23   7450  260.652174
    976   2016-03-23    923  771.304348
    977   2016-03-23    794  783.565217
    978   2016-03-23   1103  700.869565
    979   2016-03-23    940  903.347826
    981   2016-03-23   2628  501.260870
    983   2016-03-23   1021  873.391304
    984   2016-03-23   2653  490.347826
    985   2016-03-23  31568   19.000000
    986   2016-03-23   5017  107.695652
    988   2016-03-24   1896  489.869565
    989   2016-03-24   1529  561.956522
    990   2016-03-24   1173  828.739130
    991   2016-03-24   5589  160.217391
    992   2016-03-24   1154  803.000000
    994   2016-03-24   1359  762.956522
    995   2016-03-24   1121  861.652174
    996   2016-03-24   2612  473.391304
    997   2016-03-24   1253  888.695652
    998   2016-03-24   3124  471.000000
    999   2016-03-24  35771   20.521739
    1000  2016-03-24   6328  181.391304
    1002  2016-03-25   1904  473.347826
    1003  2016-03-25   1733  611.086957
    1004  2016-03-25   1114  798.739130
    1005  2016-03-25   4362  229.260870
    1006  2016-03-25   1125  812.782609
    1007  2016-03-25   1204  797.608696
    1008  2016-03-25   1009  896.173913
    1009  2016-03-25   3617  485.086957
    1010  2016-03-25   1266  919.304348
    1011  2016-03-25   2958  496.478261
    1012  2016-03-25  35380   21.826087
    1013  2016-03-25   5478  211.782609
    1015  2016-03-26   1441  459.695652
    1016  2016-03-26   1508  559.478261
    1017  2016-03-26    708  794.565217
    1018  2016-03-26   2858  282.521739
    1019  2016-03-26    840  800.434783
    1020  2016-03-26    856  825.086957
    1021  2016-03-26   2804  946.826087
    1022  2016-03-26   2284  438.782609
    1023  2016-03-26    859  881.695652
    1024  2016-03-26   2744  503.478261
    1025  2016-03-26  28448   20.869565
    1026  2016-03-26   3938  234.173913
    1028  2016-03-27   1417  467.130435
    1029  2016-03-27   1104  515.521739
    1030  2016-03-27    740  846.000000
    1031  2016-03-27   2453  349.304348
    1032  2016-03-27    853  800.478261
    1033  2016-03-27    856  846.521739
    1034  2016-03-27   3839  412.391304
    1035  2016-03-27   2192  464.434783
    1036  2016-03-27    778  906.130435
    1037  2016-03-27   2786  487.391304
    1038  2016-03-27  30854   19.956522
    1039  2016-03-27   3834  248.913043
    1041  2016-03-28   1489  483.434783
    1042  2016-03-28    946  585.608696
    1043  2016-03-28    750  852.304348
    1044  2016-03-28   2510  406.826087
    1045  2016-03-28    914  795.347826
    1046  2016-03-28    847  891.434783
    1047  2016-03-28   4023  242.913043
    1048  2016-03-28   3930  424.260870
    1050  2016-03-28   3276  451.608696
    1051  2016-03-28  29605   17.652174
    1052  2016-03-28   3952  247.347826
    1055  2016-03-29   1457  493.347826
    1056  2016-03-29   1060  668.608696
    1057  2016-03-29    963  856.000000
    1058  2016-03-29   1390  449.217391
    1059  2016-03-29    998  788.086957
    1060  2016-03-29    924  931.260870
    1061  2016-03-29   1689  233.391304
    1063  2016-03-29   3328  315.000000
    1065  2016-03-29   3006  411.000000
    1066  2016-03-29  30412   16.826087
    1067  2016-03-29   3748  264.130435
    1070  2016-03-30   2236  494.391304
    1071  2016-03-30   1891  661.130435
    1072  2016-03-30   1497  736.043478
    1073  2016-03-30   1426  595.478261
    1074  2016-03-30   2628  707.956522
    1076  2016-03-30   1039  945.086957
    1077  2016-03-30   1117  431.086957
    1078  2016-03-30   1364  550.826087
    1079  2016-03-30   2926  337.086957
    1081  2016-03-30   1297  917.565217
    1082  2016-03-30   3257  455.391304
    1085  2016-03-30   3859  269.000000
    1088  2016-03-31   2041  445.000000
    1089  2016-03-31   1526  506.130435
    1090  2016-03-31   1157  622.652174
    1091  2016-03-31   1388  673.521739
    1092  2016-03-31   1800  500.913043
    1093  2016-03-31   2433  954.826087
    1094  2016-03-31   1022  905.434783
    1095  2016-03-31   1223  625.434783
    1096  2016-03-31    237  579.869565
    1097  2016-03-31  11099  389.565217
    1098  2016-03-31   1372  918.217391
    1099  2016-03-31   3294  479.130435
    1100  2016-03-31  35903   21.739130
    1101  2016-03-31    777  944.652174
    1102  2016-03-31   3770  312.956522
    


```python
newData1.columns
```




    Index(['Date', 'sales', 'avgrank'], dtype='object')




```python
newData1.Date.dtype
```




    dtype('O')




```python
pd.options.mode.chained_assignment = None  # default='warn'

```

### Data Formatting:

As we can see that there is a date column in our dataframe but it is of type 'O' . So we will convert the data type and date & time format so that it can be interpreted as a time series data by our model. We will also change the index of the dataframe to Date.


```python
newData1['Date'] = pd.to_datetime(newData1['Date'] , format = '%Y/%m/%d')
newData2 = newData1.drop(['Date'], axis=1)
newData2.index = newData.Date
```


```python
print(newData2)
```

                sales     avgrank
    Date                         
    2016-01-01   2412  469.739130
    2016-01-01   1308  719.173913
    2016-01-01   2037  603.695652
    2016-01-01   2052  567.913043
    2016-01-01   1553  776.826087
    2016-01-01   2152  518.260870
    2016-01-01  89289    5.782609
    2016-01-02   1622  464.043478
    2016-01-02   1110  754.652174
    2016-01-02   1553  586.217391
    2016-01-02   1467  617.782609
    2016-01-02   1011  727.695652
    2016-01-02   1722  546.260870
    2016-01-02  84915    5.826087
    2016-01-03   1564  508.695652
    2016-01-03    963  762.434783
    2016-01-03   1409  560.521739
    2016-01-03   1371  676.000000
    2016-01-03    938  829.217391
    2016-01-03   1662  560.913043
    2016-01-03   2029  680.086957
    2016-01-03  61345    5.000000
    2016-01-04   1609  525.695652
    2016-01-04   1168  797.217391
    2016-01-04   1732  573.826087
    2016-01-04   1421  715.043478
    2016-01-04    989  840.652174
    2016-01-04   1081  573.782609
    2016-01-04   1460  654.565217
    2016-01-04  58943    6.391304
    2016-01-05   1594  548.173913
    2016-01-05   1163  750.869565
    2016-01-05   1392  563.130435
    2016-01-05   1482  757.086957
    2016-01-05   1055  875.173913
    2016-01-05    984  743.217391
    2016-01-05   1102  807.869565
    2016-01-05  62196   10.782609
    2016-01-06   1883  545.739130
    2016-01-06   1252  742.173913
    2016-01-06   1737  706.608696
    2016-01-06   1227  855.782609
    2016-01-06   1244  848.260870
    2016-01-06  67759   10.478261
    2016-01-07   1978  527.478261
    2016-01-07   1254  763.217391
    2016-01-07   1538  701.173913
    2016-01-07   1125  845.086957
    2016-01-07   1184  878.869565
    2016-01-07    956  940.260870
    2016-01-07  60462    9.434783
    2016-01-08   1509  483.608696
    2016-01-08    960  736.043478
    2016-01-08   1273  702.695652
    2016-01-08    893  818.695652
    2016-01-08    876  899.391304
    2016-01-08  36569   11.608696
    2016-01-09   1256  502.652174
    2016-01-09    761  783.478261
    2016-01-09    867  657.913043
    2016-01-09   1038  705.956522
    2016-01-09    808  789.695652
    2016-01-09    871  872.608696
    2016-01-09  28872   16.695652
    2016-01-10   1316  519.652174
    2016-01-10    775  815.913043
    2016-01-10   1248  702.260870
    2016-01-10    795  804.869565
    2016-01-10    866  844.000000
    2016-01-10  32229   20.608696
    2016-01-11   1350  524.173913
    2016-01-11    744  832.000000
    2016-01-11   1454  430.869565
    2016-01-11   1387  651.565217
    2016-01-11   1008  857.695652
    2016-01-11    879  824.782609
    2016-01-11   1505  811.608696
    2016-01-11  30459   19.086957
    2016-01-12   1650  525.478261
    2016-01-12   1071  867.130435
    2016-01-12   1589  481.826087
    2016-01-12   1622  588.739130
    2016-01-12    909  923.565217
    2016-01-12    946  821.347826
    2016-01-12   2339  675.521739
    2016-01-12   2047  774.913043
    2016-01-12  30254   21.608696
    2016-01-14   2046  562.434783
    2016-01-14   1285  824.391304
    2016-01-14   1893  480.826087
    2016-01-14   2371  636.304348
    2016-01-14  49142   19.782609
    2016-01-15   1813  531.565217
    2016-01-15   1132  843.869565
    2016-01-15   1478  556.695652
    2016-01-15   1613  686.086957
    2016-01-15   1110  810.347826
    2016-01-15   2081  637.739130
    2016-01-15   1769  728.521739
    2016-01-15  37157   19.173913
    2016-01-16   1372  493.043478
    2016-01-16    844  834.304348
    2016-01-16   1391  685.695652
    2016-01-16    900  826.260870
    2016-01-16   1554  639.260870
    2016-01-16   1443  757.043478
    2016-01-16  30890   23.956522
    2016-01-17   1385  519.478261
    2016-01-17    892  823.521739
    2016-01-17    977  624.521739
    2016-01-17   2729  606.086957
    2016-01-17    919  828.739130
    2016-01-17   1673  661.000000
    2016-01-17   1115  787.478261
    2016-01-17  27467   25.695652
    2016-01-18   1291  546.608696
    2016-01-18    777  841.695652
    2016-01-18   3084  433.347826
    2016-01-18    804  809.391304
    2016-01-18    943  678.217391
    2016-01-18  44326   68.956522
    2016-01-18  22792   28.304348
    2016-01-19   1368  545.565217
    2016-01-19    945  879.913043
    2016-01-19   1977  733.652174
    2016-01-19   2754  383.173913
    2016-01-19    920  835.869565
    2016-01-19   1094  788.434783
    2016-01-19  47512   25.739130
    2016-01-19  29160   32.608696
    2016-01-20   1746  565.173913
    2016-01-20   1267  876.347826
    2016-01-20   3107  390.652174
    2016-01-20   1096  859.695652
    2016-01-20   1319  749.478261
    2016-01-20   1355  766.130435
    2016-01-20  49850   24.826087
    2016-01-20  36121   33.000000
    2016-01-21   1828  555.695652
    2016-01-21   1124  805.956522
    2016-01-21   1923  401.782609
    2016-01-21   2897  407.304348
    2016-01-21   1220  848.739130
    2016-01-21   1399  714.043478
    2016-01-21   1582  854.217391
    2016-01-21   1958  838.695652
    2016-01-21  45371   27.652174
    2016-01-21  34248   35.478261
    2016-01-22   1232  514.913043
    2016-01-22    727  821.478261
    2016-01-22   1532  462.260870
    2016-01-22   2090  431.043478
    2016-01-22    842  768.130435
    2016-01-22   1043  692.086957
    2016-01-22   1284  892.043478
    2016-01-22   1654  667.043478
    2016-01-22  28863   29.000000
    2016-01-22  21218   35.913043
    2016-01-23   1167  502.173913
    2016-01-23    798  847.739130
    2016-01-23   1198  479.521739
    2016-01-23   2135  433.043478
    2016-01-23    775  769.173913
    2016-01-23    856  679.826087
    2016-01-23   1276  794.826087
    2016-01-23   1192  629.391304
    2016-01-24   1189  532.695652
    2016-01-24    708  862.695652
    2016-01-24   2028  474.652174
    2016-01-24   1781  450.130435
    2016-01-24    794  802.521739
    2016-01-24    886  737.347826
    2016-01-24   1120  750.782609
    2016-01-24   1168  645.434783
    2016-01-24  21527   36.782609
    2016-01-24  18949   32.826087
    2016-01-25   1250  539.304348
    2016-01-25    779  894.130435
    2016-01-25   1336  423.304348
    2016-01-25   1703  501.217391
    2016-01-25   1019  928.956522
    2016-01-25    791  870.000000
    2016-01-25    935  732.086957
    2016-01-25   1087  811.565217
    2016-01-25   1259  726.565217
    2016-01-25  16278   46.739130
    2016-01-25  20721   36.869565
    2016-01-26   1368  545.782609
    2016-01-26    924  902.130435
    2016-01-26   1431  520.347826
    2016-01-26   2316  507.782609
    2016-01-26    949  932.826087
    2016-01-26    931  843.956522
    2016-01-26   1029  723.521739
    2016-01-26   1204  702.347826
    2016-01-26  16522   63.478261
    2016-01-26  27454   33.217391
    2016-01-27   1807  553.434783
    2016-01-27   1317  874.086957
    2016-01-27   2079  520.347826
    2016-01-27   2487  482.565217
    2016-01-27   1157  853.130435
    2016-01-27   1213  757.086957
    2016-01-27   1758  746.652174
    2016-01-27  18981   77.739130
    2016-01-27  34159   31.391304
    2016-01-28   1639  554.608696
    2016-01-28   1188  799.391304
    2016-01-28   2361  514.521739
    2016-01-28   2116  505.304348
    2016-01-28   1147  840.000000
    2016-01-28   1321  781.956522
    2016-01-28   1935  726.347826
    2016-01-28  17933   85.260870
    2016-01-28  34537   30.826087
    2016-01-29   1240  537.173913
    2016-01-29    703  816.304348
    2016-01-29   1480  456.391304
    2016-01-29   1502  540.739130
    2016-01-29    827  842.521739
    2016-01-29    863  748.565217
    2016-01-29   7055  164.739130
    2016-01-29    386  736.652174
    2016-01-29  13235   88.956522
    2016-01-29  20230   30.913043
    2016-01-30   1175  529.652174
    2016-01-30    667  868.956522
    2016-01-30   1409  467.521739
    2016-01-30   1727  564.826087
    2016-01-30    797  824.173913
    2016-01-30   1197  789.565217
    2016-01-30   7829  138.565217
    2016-01-30  13021   87.304348
    2016-01-30  16766   38.478261
    2016-01-31   1149  532.739130
    2016-01-31    711  896.478261
    2016-01-31   1207  495.521739
    2016-01-31   2059  475.086957
    2016-01-31    790  781.086957
    2016-01-31    739  676.043478
    2016-01-31   6435  128.565217
    2016-01-31  11656   91.043478
    2016-01-31  15648   45.173913
    2016-02-01   1270  547.086957
    2016-02-01    773  884.434783
    2016-02-01   1271  525.869565
    2016-02-01   1649  473.521739
    2016-02-01    928  777.652174
    2016-02-01    751  772.565217
    2016-02-01   6630  134.260870
    2016-02-01   8485  105.260870
    2016-02-01  16267   50.565217
    2016-02-02   1435  549.434783
    2016-02-02   1003  873.434783
    2016-02-02   1181  588.391304
    2016-02-02   1770  505.000000
    2016-02-02   1080  747.652174
    2016-02-02   4731  155.739130
    2016-02-02   8677  145.565217
    2016-02-02  18581   56.521739
    2016-02-03   1858  563.478261
    2016-02-03   1419  817.521739
    2016-02-03   1409  630.000000
    2016-02-03   2330  540.521739
    2016-02-03   1361  768.043478
    2016-02-03   2569  765.043478
    2016-02-03   6081  197.130435
    2016-02-03   9948  163.565217
    2016-02-03  25123   57.173913
    2016-02-04   1869  537.782609
    2016-02-04   1312  813.956522
    2016-02-04   1678  648.086957
    2016-02-04   2164  550.304348
    2016-02-04   1254  737.217391
    2016-02-04   3304  621.826087
    2016-02-04   4468  228.086957
    2016-02-04   9212  183.956522
    2016-02-04  23394   59.565217
    2016-02-04   6235  262.086957
    2016-02-05   1320  520.086957
    2016-02-05    846  800.695652
    2016-02-05   1293  616.869565
    2016-02-05   1449  561.782609
    2016-02-05    925  754.173913
    2016-02-05   2616  541.304348
    2016-02-05   2302  293.695652
    2016-02-05   6666  186.130435
    2016-02-05  14004   60.043478
    2016-02-05   3760  248.391304
    2016-02-06   1280  533.695652
    2016-02-06    769  808.304348
    2016-02-06   1014  612.695652
    2016-02-06   1769  557.695652
    2016-02-06    925  759.521739
    2016-02-06   2343  530.391304
    2016-02-06   1973  386.956522
    2016-02-06   6154  177.565217
    2016-02-06  13272   65.478261
    2016-02-06   3408  295.260870
    2016-02-06   3552  364.260870
    2016-02-07   1439  536.434783
    2016-02-07    929  850.434783
    2016-02-07    979  654.217391
    2016-02-07   1663  515.086957
    2016-02-07    989  789.565217
    2016-02-07   2189  521.739130
    2016-02-07   1938  458.608696
    2016-02-07   6039  181.217391
    2016-02-07  12973   69.869565
    2016-02-07   3955  334.391304
    2016-02-07   3062  379.000000
    2016-02-08   1359  530.869565
    2016-02-08    815  838.304348
    2016-02-08   1439  692.521739
    2016-02-08   1667  554.956522
    2016-02-08    927  768.913043
    2016-02-08   2127  516.391304
    2016-02-08   1618  515.260870
    2016-02-08   5211  193.304348
    2016-02-08  11992   78.173913
    2016-02-08   1317  335.000000
    2016-02-08   2002  435.304348
    2016-02-09   1735  521.478261
    2016-02-09   1020  816.739130
    2016-02-09   1465  580.739130
    2016-02-09   2006  607.391304
    2016-02-09   1088  752.304348
    2016-02-09   2027  548.434783
    2016-02-09   1568  558.826087
    2016-02-09   5879  210.043478
    2016-02-09  15179   78.130435
    2016-02-09    556  606.913043
    2016-02-09   2652  537.695652
    2016-02-10   2212  527.739130
    2016-02-10   1385  764.434783
    2016-02-10   1697  581.565217
    2016-02-10   2673  575.869565
    2016-02-10   1448  762.478261
    2016-02-10   3024  627.956522
    2016-02-10   1736  650.608696
    2016-02-10   6547  229.260870
    2016-02-10  20504   80.869565
    2016-02-10   2061  569.391304
    2016-02-11   2064  481.347826
    2016-02-11   1238  785.565217
    2016-02-11   1760  622.608696
    2016-02-11   2236  552.000000
    2016-02-11   1381  714.173913
    2016-02-11   3513  602.695652
    2016-02-11   1719  707.347826
    2016-02-11   6110  261.956522
    2016-02-11  17485   76.695652
    2016-02-11   1916  794.652174
    2016-02-12   1449  467.000000
    2016-02-12    791  825.173913
    2016-02-12   1235  604.130435
    2016-02-12   1489  593.043478
    2016-02-12   1001  709.000000
    2016-02-12   2534  523.347826
    2016-02-12   1232  723.000000
    2016-02-12   4729  272.521739
    2016-02-12  11147   79.086957
    2016-02-13   1342  487.478261
    2016-02-13    702  858.956522
    2016-02-13   1728  563.434783
    2016-02-13   1380  606.695652
    2016-02-13   1009  678.956522
    2016-02-13   2332  462.000000
    2016-02-13   1271  739.304348
    2016-02-13   4395  261.391304
    2016-02-13   9598   82.260870
    2016-02-14   2928  223.304348
    2016-02-14   1280  506.347826
    2016-02-14    674  904.260870
    2016-02-14    949  473.565217
    2016-02-14   1353  625.956522
    2016-02-14    746  924.086957
    2016-02-14    877  684.695652
    2016-02-14   1954  451.130435
    2016-02-14    826  742.739130
    2016-02-14   1090  701.347826
    2016-02-14   4079  261.043478
    2016-02-14  10271   87.956522
    2016-02-15   2029  278.826087
    2016-02-15   1430  485.391304
    2016-02-15    768  902.086957
    2016-02-15    940  562.173913
    2016-02-15   1441  606.608696
    2016-02-15    969  692.000000
    2016-02-15   2134  479.173913
    2016-02-15   1316  726.478261
    2016-02-15   4924  253.956522
    2016-02-15  11795   78.000000
    2016-02-16   1851  415.913043
    2016-02-16   1582  480.956522
    2016-02-16   1000  888.434783
    2016-02-16   1409  644.608696
    2016-02-16   1997  598.000000
    2016-02-16   1079  726.913043
    2016-02-16   1562  867.652174
    2016-02-16   2062  503.173913
    2016-02-16   1478  710.826087
    2016-02-16   5353  251.782609
    2016-02-16  15537   75.478261
    2016-02-17   1723  561.956522
    2016-02-17   1919  503.304348
    2016-02-17   1307  865.000000
    2016-02-17   1661  582.913043
    2016-02-17   2217  562.391304
    2016-02-17   1327  750.565217
    2016-02-17   1960  807.521739
    2016-02-17   2706  569.739130
    2016-02-17   1865  747.043478
    2016-02-17   8867  265.608696
    2016-02-17  20417   74.652174
    2016-02-18   1419  705.086957
    2016-02-18   2025  506.782609
    2016-02-18   1293  802.217391
    2016-02-18   1821  584.217391
    2016-02-18   2385  574.869565
    2016-02-18   1310  771.304348
    2016-02-18   1734  835.739130
    2016-02-18   3556  552.391304
    2016-02-18   2286  733.695652
    2016-02-18  35279  184.565217
    2016-02-18  20284   72.043478
    2016-02-19   1357  798.869565
    2016-02-19   1560  505.173913
    2016-02-19   1021  810.000000
    2016-02-19    715  628.000000
    2016-02-19   1969  533.086957
    2016-02-19   1170  775.478261
    2016-02-19   1476  859.521739
    2016-02-19   3099  496.043478
    2016-02-19   1933  668.304348
    2016-02-19  16330   78.826087
    2016-02-19  19032   71.130435
    2016-02-20   1302  513.260870
    2016-02-20    856  844.695652
    2016-02-20   1271  794.260870
    2016-02-20   1555  542.043478
    2016-02-20    946  780.913043
    2016-02-20   1163  839.000000
    2016-02-20   3295  443.869565
    2016-02-20   1310  670.000000
    2016-02-20  10265   98.652174
    2016-02-20  14375   64.173913
    2016-02-21   1357  508.869565
    2016-02-21    799  826.521739
    2016-02-21   1187  679.913043
    2016-02-21   1413  560.130435
    2016-02-21    972  771.695652
    2016-02-21   1138  880.826087
    2016-02-21   3777  381.652174
    2016-02-21   1256  706.695652
    2016-02-21   8114  119.260870
    2016-02-21  14902   63.478261
    2016-02-22   1396  508.086957
    2016-02-22    789  821.217391
    2016-02-22   1086  623.347826
    2016-02-22   1486  600.391304
    2016-02-22   1024  735.217391
    2016-02-22    806  882.956522
    2016-02-22   5447  321.782609
    2016-02-22    820  657.739130
    2016-02-22   1297  741.391304
    2016-02-22   7352  150.478261
    2016-02-22  18591   60.608696
    2016-02-23   1654  499.739130
    2016-02-23   1059  848.130435
    2016-02-23   1381  589.217391
    2016-02-23   1763  610.130435
    2016-02-23   1149  717.173913
    2016-02-23   3896  349.652174
    2016-02-23   1469  736.043478
    2016-02-23   7706  163.869565
    2016-02-23  21589   45.000000
    2016-02-24   2128  492.913043
    2016-02-24   1373  777.217391
    2016-02-24   1614  581.608696
    2016-02-24   2165  589.608696
    2016-02-24   1373  706.869565
    2016-02-24   3034  431.347826
    2016-02-24   2178  718.782609
    2016-02-24   8076  187.347826
    2016-02-24  29501   40.173913
    2016-02-25   1936  477.739130
    2016-02-25   1227  760.391304
    2016-02-25   1717  638.000000
    2016-02-25    983  933.434783
    2016-02-25   1981  595.086957
    2016-02-25   1319  734.652174
    2016-02-25   3048  529.739130
    2016-02-25   1689  693.478261
    2016-02-25   5546  221.956522
    2016-02-25   1647  619.304348
    2016-02-25  31090   35.304348
    2016-02-26   1480  468.173913
    2016-02-26    811  774.130435
    2016-02-26   1055  620.739130
    2016-02-26    627  925.173913
    2016-02-26   1183  612.086957
    2016-02-26    962  725.130435
    2016-02-26   2712  509.913043
    2016-02-26   1020  762.695652
    2016-02-26   4138  266.304348
    2016-02-26    176  648.521739
    2016-02-26  16154   34.521739
    2016-02-27   1458  422.956522
    2016-02-27    700  785.695652
    2016-02-27    865  646.869565
    2016-02-27    563  934.956522
    2016-02-27   1314  668.826087
    2016-02-27    892  696.913043
    2016-02-27   2770  426.565217
    2016-02-27   1199  781.130435
    2016-02-27   4485  271.695652
    2016-02-27  15148   41.130435
    2016-02-28   1430  414.217391
    2016-02-28    768  803.000000
    2016-02-28   1364  670.347826
    2016-02-28   1411  618.869565
    2016-02-28    951  708.826087
    2016-02-28   2600  374.608696
    2016-02-28   1438  758.260870
    2016-02-28   4460  260.695652
    2016-02-28  15738   43.478261
    2016-02-28   1047  880.260870
    2016-03-01   1586  423.913043
    2016-03-01    853  796.217391
    2016-03-01   1590  560.869565
    2016-03-01    662  957.086957
    2016-03-01   3075  571.608696
    2016-03-01    979  683.739130
    2016-03-01   1548  398.913043
    2016-03-01   1401  706.304348
    2016-03-01   5233  245.565217
    2016-03-01  19151   39.782609
    2016-03-01   1200  745.434783
    2016-03-02   2041  417.260870
    2016-03-02   1012  813.565217
    2016-03-02   1786  506.043478
    2016-03-02    823  928.217391
    2016-03-02   4436  431.434783
    2016-03-02   1114  681.217391
    2016-03-02   4691  477.260870
    2016-03-02   1542  537.043478
    2016-03-02   2221  635.086957
    2016-03-02   5296  223.304348
    2016-03-02  25378   34.826087
    2016-03-02   1252  663.565217
    2016-03-03   2322  393.304348
    2016-03-03   1277  805.869565
    2016-03-03   1508  524.391304
    2016-03-03   1049  924.826087
    2016-03-03   5262  371.695652
    2016-03-03   1294  716.826087
    2016-03-03   1317  872.565217
    2016-03-03   6659  327.434783
    2016-03-03   1706  683.217391
    2016-03-03   2198  505.304348
    2016-03-03   5575  253.652174
    2016-03-03  33408   29.565217
    2016-03-03   1982  662.434783
    2016-03-04   2084  404.826087
    2016-03-04   1123  801.304348
    2016-03-04   1720  594.521739
    2016-03-04    990  915.913043
    2016-03-04   4766  364.130435
    2016-03-04   1396  752.217391
    2016-03-04   1115  904.434783
    2016-03-04   6417  253.739130
    2016-03-04   1925  776.739130
    2016-03-04   1691  573.000000
    2016-03-04   5401  291.782609
    2016-03-04  32011   30.565217
    2016-03-04   2568  612.521739
    2016-03-05   1581  405.043478
    2016-03-05    664  823.173913
    2016-03-05   1888  566.000000
    2016-03-05    631  892.000000
    2016-03-05   3108  368.608696
    2016-03-05   3348  572.695652
    2016-03-05   1131  890.869565
    2016-03-05   4743  255.695652
    2016-03-05   1862  716.565217
    2016-03-05    937  695.913043
    2016-03-05   3862  293.869565
    2016-03-05  32532   28.869565
    2016-03-05   2426  496.521739
    2016-03-06   1527  410.434783
    2016-03-06    757  886.260870
    2016-03-06   1526  437.260870
    2016-03-06    786  832.913043
    2016-03-06   3066  382.000000
    2016-03-06    928  922.652174
    2016-03-06   1710  350.260870
    2016-03-06   1391  779.956522
    2016-03-06   2689  246.260870
    2016-03-06   1827  603.695652
    2016-03-06    990  778.913043
    2016-03-06   4908  289.695652
    2016-03-06  33458   17.130435
    2016-03-06   1779  412.652174
    2016-03-07   1603  431.521739
    2016-03-07    798  928.130435
    2016-03-07   1207  496.043478
    2016-03-07   1008  780.260870
    2016-03-07   3151  398.304348
    2016-03-07   1368  425.695652
    2016-03-07   1529  659.130435
    2016-03-07   1731  354.260870
    2016-03-07   2077  595.000000
    2016-03-07   1188  812.000000
    2016-03-07   4436  257.869565
    2016-03-07  34919   14.521739
    2016-03-07   1640  447.478261
    2016-03-08   1469  422.608696
    2016-03-08    711  898.434783
    2016-03-08   1096  612.956522
    2016-03-08    757  702.347826
    2016-03-08   2009  415.086957
    2016-03-08   1082  553.043478
    2016-03-08   1259  593.217391
    2016-03-08   2015  562.565217
    2016-03-08   2057  536.130435
    2016-03-08   1143  768.130435
    2016-03-08   3709  267.260870
    2016-03-08  33293   13.652174
    2016-03-08   1551  504.565217
    2016-03-09   1639  436.086957
    2016-03-09    957  930.086957
    2016-03-09   1567  623.260870
    2016-03-09    859  779.782609
    2016-03-09   1851  507.869565
    2016-03-09   1259  628.869565
    2016-03-09   1520  613.260870
    2016-03-09   2084  595.000000
    2016-03-09   1915  497.652174
    2016-03-09   1290  829.478261
    2016-03-09   3816  303.826087
    2016-03-09  35441   13.565217
    2016-03-09   2183  473.173913
    2016-03-10   1919  475.391304
    2016-03-10   1280  902.000000
    2016-03-10   1869  532.521739
    2016-03-10   1209  801.130435
    2016-03-10   2006  518.043478
    2016-03-10   1477  645.000000
    2016-03-10    998  877.217391
    2016-03-10   1766  607.434783
    2016-03-10  19010  135.608696
    2016-03-10   2478  618.000000
    2016-03-10   2616  571.478261
    2016-03-10   1957  774.956522
    2016-03-10   4539  333.391304
    2016-03-10  41386   15.826087
    2016-03-10   2946  408.347826
    2016-03-12   1395  481.565217
    2016-03-12    895  906.260870
    2016-03-12   1154  575.956522
    2016-03-12    919  752.565217
    2016-03-12   1231  650.043478
    2016-03-12    874  730.173913
    2016-03-12   1602  657.086957
    2016-03-12  16656   45.130435
    2016-03-12   1771  557.260870
    2016-03-12   2661  457.869565
    2016-03-12    993  873.608696
    2016-03-12   3257  365.086957
    2016-03-12  28069   23.000000
    2016-03-12   1447  404.913043
    2016-03-13   1772  477.130435
    2016-03-13    833  891.086957
    2016-03-13   1127  615.434783
    2016-03-13   1036  747.739130
    2016-03-13   1188  691.826087
    2016-03-13    898  764.565217
    2016-03-13   2082  442.956522
    2016-03-13   1502  599.130435
    2016-03-13   4107   52.217391
    2016-03-13   1385  595.260870
    2016-03-13   2750  439.130435
    2016-03-13   1027  931.608696
    2016-03-13   2941  371.608696
    2016-03-13  28991   22.826087
    2016-03-13    872  531.130435
    2016-03-14   1528  431.565217
    2016-03-14    738  921.869565
    2016-03-14   1084  654.652174
    2016-03-14    871  783.043478
    2016-03-14   1163  734.521739
    2016-03-14    880  771.608696
    2016-03-14   1433  507.608696
    2016-03-14   1039  607.478261
    2016-03-14   1761  116.565217
    2016-03-14   1777  683.217391
    2016-03-14   2917  411.739130
    2016-03-14    909  891.173913
    2016-03-14   3078  405.478261
    2016-03-14  27801   20.782609
    2016-03-14    815  710.913043
    2016-03-15   1614  416.173913
    2016-03-15    758  931.260870
    2016-03-15    873  689.608696
    2016-03-15    743  778.521739
    2016-03-15   1139  751.695652
    2016-03-15    953  764.782609
    2016-03-15   1653  639.478261
    2016-03-15    787  708.260870
    2016-03-15    443  622.652174
    2016-03-15    472  268.913043
    2016-03-15   1107  634.000000
    2016-03-15   2533  398.782609
    2016-03-15   1234  912.608696
    2016-03-15   3078  385.000000
    2016-03-15  27124   18.869565
    2016-03-15    674  854.347826
    2016-03-16   1639  432.000000
    2016-03-16   1088  981.000000
    2016-03-16   1643  816.000000
    2016-03-16   1036  822.000000
    2016-03-16   1288  778.000000
    2016-03-16    929  757.000000
    2016-03-16   1993  632.000000
    2016-03-16    779  824.000000
    2016-03-16    151  826.000000
    2016-03-16     48  463.000000
    2016-03-16    913  725.000000
    2016-03-16   2490  429.000000
    2016-03-16   2112  807.000000
    2016-03-16   3049  402.000000
    2016-03-16  30309   20.000000
    2016-03-16    704  957.000000
    2016-03-16  23134  121.000000
    2016-03-17   1826  471.521739
    2016-03-17   1266  882.695652
    2016-03-17   1674  632.869565
    2016-03-17   1286  758.347826
    2016-03-17   1518  778.347826
    2016-03-17   1170  786.434783
    2016-03-17   2531  593.826087
    2016-03-17   1025  929.652174
    2016-03-17   2808  477.695652
    2016-03-17   1842  602.391304
    2016-03-17   3495  426.434783
    2016-03-17  34961   20.478261
    2016-03-17  26072   70.565217
    2016-03-18   1680  499.434783
    2016-03-18   1114  894.434783
    2016-03-18   1160  569.391304
    2016-03-18   1154  719.695652
    2016-03-18   1367  788.956522
    2016-03-18   1143  811.913043
    2016-03-18   2260  539.130435
    2016-03-18   2897  522.086957
    2016-03-18   1793  636.826087
    2016-03-18   3170  468.000000
    2016-03-18  32913   23.652174
    2016-03-18  20929   49.913043
    2016-03-19   1325  513.260870
    2016-03-19    698  705.956522
    2016-03-19    661  746.869565
    2016-03-19   1070  786.956522
    2016-03-19    902  790.000000
    2016-03-19   1566  546.304348
    2016-03-19   2300  477.608696
    2016-03-19   1238  618.739130
    2016-03-19   3016  451.391304
    2016-03-19  24902   23.217391
    2016-03-19  15211   55.956522
    2016-03-20   1314  507.260870
    2016-03-20    985  802.913043
    2016-03-20    764  802.173913
    2016-03-20    997  814.565217
    2016-03-20    840  805.304348
    2016-03-20   1505  538.521739
    2016-03-20   2374  474.173913
    2016-03-20   1297  646.521739
    2016-03-20   1352  680.956522
    2016-03-20   2679  407.739130
    2016-03-20  29736   23.217391
    2016-03-20  15297   57.391304
    2016-03-21   1569  523.521739
    2016-03-21    786  747.565217
    2016-03-21    849  821.391304
    2016-03-21   2474  823.260870
    2016-03-21    973  830.086957
    2016-03-21   1522  606.086957
    2016-03-21   2116  493.695652
    2016-03-21   1069  708.565217
    2016-03-21   1135  715.739130
    2016-03-21   2648  422.913043
    2016-03-21  29122   17.043478
    2016-03-21  15809   59.565217
    2016-03-22   1392  487.869565
    2016-03-22   1920  850.130435
    2016-03-22    752  812.000000
    2016-03-22  12717  676.130435
    2016-03-22    896  773.695652
    2016-03-22   1192  621.913043
    2016-03-22    726  883.347826
    2016-03-22    892  867.826087
    2016-03-22   2114  528.086957
    2016-03-22    885  785.565217
    2016-03-22    934  775.521739
    2016-03-22   2740  472.521739
    2016-03-22  27754   18.913043
    2016-03-22   6649   60.086957
    2016-03-23   1514  486.130435
    2016-03-23   1469  561.260870
    2016-03-23    860  837.043478
    2016-03-23   7450  260.652174
    2016-03-23    923  771.304348
    2016-03-23    794  783.565217
    2016-03-23   1103  700.869565
    2016-03-23    940  903.347826
    2016-03-23   2628  501.260870
    2016-03-23   1021  873.391304
    2016-03-23   2653  490.347826
    2016-03-23  31568   19.000000
    2016-03-23   5017  107.695652
    2016-03-24   1896  489.869565
    2016-03-24   1529  561.956522
    2016-03-24   1173  828.739130
    2016-03-24   5589  160.217391
    2016-03-24   1154  803.000000
    2016-03-24   1359  762.956522
    2016-03-24   1121  861.652174
    2016-03-24   2612  473.391304
    2016-03-24   1253  888.695652
    2016-03-24   3124  471.000000
    2016-03-24  35771   20.521739
    2016-03-24   6328  181.391304
    2016-03-25   1904  473.347826
    2016-03-25   1733  611.086957
    2016-03-25   1114  798.739130
    2016-03-25   4362  229.260870
    2016-03-25   1125  812.782609
    2016-03-25   1204  797.608696
    2016-03-25   1009  896.173913
    2016-03-25   3617  485.086957
    2016-03-25   1266  919.304348
    2016-03-25   2958  496.478261
    2016-03-25  35380   21.826087
    2016-03-25   5478  211.782609
    2016-03-26   1441  459.695652
    2016-03-26   1508  559.478261
    2016-03-26    708  794.565217
    2016-03-26   2858  282.521739
    2016-03-26    840  800.434783
    2016-03-26    856  825.086957
    2016-03-26   2804  946.826087
    2016-03-26   2284  438.782609
    2016-03-26    859  881.695652
    2016-03-26   2744  503.478261
    2016-03-26  28448   20.869565
    2016-03-26   3938  234.173913
    2016-03-27   1417  467.130435
    2016-03-27   1104  515.521739
    2016-03-27    740  846.000000
    2016-03-27   2453  349.304348
    2016-03-27    853  800.478261
    2016-03-27    856  846.521739
    2016-03-27   3839  412.391304
    2016-03-27   2192  464.434783
    2016-03-27    778  906.130435
    2016-03-27   2786  487.391304
    2016-03-27  30854   19.956522
    2016-03-27   3834  248.913043
    2016-03-28   1489  483.434783
    2016-03-28    946  585.608696
    2016-03-28    750  852.304348
    2016-03-28   2510  406.826087
    2016-03-28    914  795.347826
    2016-03-28    847  891.434783
    2016-03-28   4023  242.913043
    2016-03-28   3930  424.260870
    2016-03-28   3276  451.608696
    2016-03-28  29605   17.652174
    2016-03-28   3952  247.347826
    2016-03-29   1457  493.347826
    2016-03-29   1060  668.608696
    2016-03-29    963  856.000000
    2016-03-29   1390  449.217391
    2016-03-29    998  788.086957
    2016-03-29    924  931.260870
    2016-03-29   1689  233.391304
    2016-03-29   3328  315.000000
    2016-03-29   3006  411.000000
    2016-03-29  30412   16.826087
    2016-03-29   3748  264.130435
    2016-03-30   2236  494.391304
    2016-03-30   1891  661.130435
    2016-03-30   1497  736.043478
    2016-03-30   1426  595.478261
    2016-03-30   2628  707.956522
    2016-03-30   1039  945.086957
    2016-03-30   1117  431.086957
    2016-03-30   1364  550.826087
    2016-03-30   2926  337.086957
    2016-03-30   1297  917.565217
    2016-03-30   3257  455.391304
    2016-03-30   3859  269.000000
    2016-03-31   2041  445.000000
    2016-03-31   1526  506.130435
    2016-03-31   1157  622.652174
    2016-03-31   1388  673.521739
    2016-03-31   1800  500.913043
    2016-03-31   2433  954.826087
    2016-03-31   1022  905.434783
    2016-03-31   1223  625.434783
    2016-03-31    237  579.869565
    2016-03-31  11099  389.565217
    2016-03-31   1372  918.217391
    2016-03-31   3294  479.130435
    2016-03-31  35903   21.739130
    2016-03-31    777  944.652174
    2016-03-31   3770  312.956522
    

### Training Data:

We will now extract the 80% values of this new dataframe and treat them as training data values.


```python
train = newData2[:int(0.8*(len(newData2)))]

```


```python
import matplotlib.pyplot as plt

```

Now let's plot the sales values so that it gives us an idea of how widely spread the sales values are. It gives us an insight on the range of values for sales. 


```python
newData2['sales'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15b73ee31d0>




![png](output_40_1.png)


### Check for Time Series Stationarity:

In time series analysis and forecasting, it is important for us to know if the time series in stationary or not. This can be done by plotting a graph and more precisely by running some statistical tests.

There are many tests which can tell you about the stationarity of the time series data, like ADF test, KPSS test etc. We have used an ADF (Dickey-Fuller) test to determine this.


```python
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

#apply adf test on the series
adf_test(train['sales'])
```

    Results of Dickey-Fuller Test:
    Test Statistic                  -4.060158
    p-value                          0.001126
    #Lags Used                      18.000000
    Number of Observations Used    716.000000
    Critical Value (1%)             -3.439516
    Critical Value (5%)             -2.865585
    Critical Value (10%)            -2.568924
    dtype: float64
    

### Note : 
In this test, as the Test Statistic is less than the critical value, we can reject the null hypothesis, which also means that the series is stationary. If the test statistic was higher than the critical value, then we cannot reject the null hypothesis and the time series in that case will be non-stationary. Differencing is required to convert a non stationary time series into a stationary one. 

### Vector Autoregression (VAR) Model:

It is important for us to understand which model to be used with respect to the problem statement. In our case, we should notice that the time series is not a univariate time series but a multivariate time series. In such time series, there are more than one variables to be fed in the model as input. As we know here, Sales is a function of Time and Ranks, so this problem cannot be solved using ARIMA model. Instead, we will use a Vector Autoregression Model and train it to understand that Sales is a function of Time and Rank.


```python
from statsmodels.tsa.vector_ar.var_model import VAR
model = VAR(endog=train)
results = model.fit()
```

    C:\Users\dhira\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:225: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    


```python
results.summary()
```




      Summary of Regression Results   
    ==================================
    Model:                         VAR
    Method:                        OLS
    Date:           Wed, 21, Nov, 2018
    Time:                     05:58:08
    --------------------------------------------------------------------
    No. of Equations:         2.00000    BIC:                    28.9941
    Nobs:                     734.000    HQIC:                   28.9710
    Log likelihood:          -12704.0    FPE:                3.76394e+12
    AIC:                      28.9565    Det(Omega_mle):     3.73336e+12
    --------------------------------------------------------------------
    Results for equation sales
    =============================================================================
                    coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------
    const           7188.095553      1327.119449            5.416           0.000
    L1.sales          -0.013136         0.050402           -0.261           0.794
    L1.avgrank        -3.750460         1.978577           -1.896           0.058
    =============================================================================
    
    Results for equation avgrank
    =============================================================================
                    coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------
    const            499.568193        33.416604           14.950           0.000
    L1.sales          -0.002050         0.001269           -1.616           0.106
    L1.avgrank         0.107571         0.049820            2.159           0.031
    =============================================================================
    
    Correlation matrix of residuals
                  sales   avgrank
    sales      1.000000 -0.681426
    avgrank   -0.681426  1.000000
    
    



### Plot the model_fit object:

Lets have a look at the variance of the sales and ranks data. These plots are generated from the interpretation of the training data by VAR model.


```python
results.plot()
```




![png](output_48_0.png)




![png](output_48_1.png)


Lets plot the auto-correlation function to understand the correlation for time series observations with respect to the previous time steps, called lags. 



```python
results.plot_acorr()
```




![png](output_50_0.png)




![png](output_50_1.png)


As we can see in the plot, the values of y-axis are the confidence intervals and the values on x-axis are the lags. The two horizontal dotted lines in a plot determines a range of the time-series values w.r.t the lags. This plot also gives us the information of the number of lags that we can use in our VAR model. In our case, we can use maxlags = 15 .


```python
 results = model.fit(maxlags=15, ic='aic')
```


```python
lag_order = results.k_ar
```

### Fit the model:

Let's fit the model on entire data set and forecast 90 days ahead of current time. We are given 3 months of data and we will predict the sales and ranks for 3 months in future.


```python
Estimates = results.forecast(newData2.values[-lag_order:], 90)
print(Estimates)

```

    [[ 2840.9997398    766.55819819]
     [-3328.33224535   768.92426941]
     [ 3442.98895128   711.5326563 ]
     [ 6847.15397636   608.70795631]
     [11409.9897393    483.837487  ]
     [13394.93730923   464.58679922]
     [ 4279.04575772   540.87150138]
     [ 8296.26502653   464.07885361]
     [ 4054.8054321    605.09264174]
     [  231.85003493   696.99709723]
     [11334.16608751   465.13579213]
     [ 5403.44923458   672.80952569]
     [ 5432.91721041   586.72654792]
     [ 8019.8436417    592.6446544 ]
     [ 4546.70601552   567.03138658]
     [ 6033.86097957   565.91417052]
     [ 5403.18085084   553.77807264]
     [ 6431.92055022   525.28791605]
     [ 8427.96098473   522.71537805]
     [ 5305.53989777   567.49954485]
     [ 6536.30019512   570.64005632]
     [ 6193.72228193   573.12941588]
     [ 3804.96923387   633.59549812]
     [ 6048.63990018   570.52243336]
     [ 5762.96746691   575.97963205]
     [ 5780.71257099   556.68051503]
     [ 6849.01283309   542.48619231]
     [ 5855.8400033    537.36308196]
     [ 6305.73027813   544.80184995]
     [ 5722.99811567   558.13645455]
     [ 5368.1139812    568.10714192]
     [ 5982.92748944   571.36146014]
     [ 5302.00933452   577.14655284]
     [ 5565.58875399   579.0085594 ]
     [ 5889.43201279   558.93154261]
     [ 5391.16869031   566.01017793]
     [ 5725.38322837   553.05561002]
     [ 5787.20073179   547.37273024]
     [ 5683.40167542   549.25198014]
     [ 5871.70661985   551.5172317 ]
     [ 5543.95618446   557.28544658]
     [ 5605.45301218   562.77021548]
     [ 5453.57236052   566.47280035]
     [ 5277.64111132   567.98414763]
     [ 5455.24004668   564.81428074]
     [ 5420.77687744   558.34029579]
     [ 5483.37308721   556.92665379]
     [ 5614.47621425   549.23832338]
     [ 5548.71973182   550.08416308]
     [ 5521.40008802   552.08441466]
     [ 5470.25830851   553.7995956 ]
     [ 5368.14962473   558.08819024]
     [ 5364.01624925   560.07640496]
     [ 5289.35440125   560.96506812]
     [ 5315.15708607   560.0126222 ]
     [ 5335.99469679   557.35228042]
     [ 5329.4711038    554.75872575]
     [ 5370.52402674   552.83070783]
     [ 5375.2815719    550.53957805]
     [ 5355.99243486   551.79089462]
     [ 5337.64801989   552.34993771]
     [ 5302.80672908   553.99701946]
     [ 5258.75902116   556.08533622]
     [ 5231.83059313   556.54143467]
     [ 5210.17096186   556.70056974]
     [ 5217.37835417   555.51318282]
     [ 5223.60186681   553.90861148]
     [ 5237.95884152   552.43010842]
     [ 5248.17030834   551.25247297]
     [ 5242.54236078   550.86511299]
     [ 5225.50938114   551.538643  ]
     [ 5203.92100952   552.11356579]
     [ 5178.01533968   553.28987422]
     [ 5153.64614402   553.94213406]
     [ 5144.24416857   553.92870515]
     [ 5137.4719728    553.6347964 ]
     [ 5139.95256492   552.68124278]
     [ 5144.52145518   551.79885458]
     [ 5146.85829413   551.00722072]
     [ 5145.82042273   550.58393884]
     [ 5137.23587609   550.64909281]
     [ 5123.84952527   551.01838741]
     [ 5109.50839633   551.47860867]
     [ 5092.83408089   551.98765892]
     [ 5081.35985583   552.10870262]
     [ 5075.40449536   551.97897494]
     [ 5071.2599925    551.59387601]
     [ 5073.11083862   550.99538795]
     [ 5073.39811032   550.50581757]
     [ 5072.09376233   550.12526407]]
    

Lets plot the forecasted values against the Observed value and Std. error.


```python
results.plot_forecast(15)
```




![png](output_57_0.png)




![png](output_57_1.png)


## Final Data:

We have set our start date to 2016-04-01 and forecasted the values for three months for two variables i.e., ranks and sales. We also appended the app id colum from the previous dataframes and the final data looks like below.


```python
dataset = pd.DataFrame({'sales_est':Estimates[:,0],'ranks_est':Estimates[:,1]})
date = np.array('2016-04-01', dtype=np.datetime64)
Futuredate = date + np.arange(90)
df = pd.DataFrame(data =Futuredate,columns=['Dates'])
df['app_id'] = salesdata['app_id'].astype(int)
df['sales_est'] = dataset['sales_est'].astype(int)
df['ranks_est'] = dataset['ranks_est'].astype(int)
print(df)


```

            Dates  app_id  sales_est  ranks_est
    0  2016-04-01     320       2840        766
    1  2016-04-02     406      -3328        768
    2  2016-04-03     459       3442        711
    3  2016-04-04     722       6847        608
    4  2016-04-05    1234      11409        483
    5  2016-04-06    1490      13394        464
    6  2016-04-07    2398       4279        540
    7  2016-04-08    2891       8296        464
    8  2016-04-09     320       4054        605
    9  2016-04-10     406        231        696
    ..        ...     ...        ...        ...
    80 2016-06-20    2891       5137        550
    81 2016-06-21     320       5123        551
    82 2016-06-22     406       5109        551
    83 2016-06-23     459       5092        551
    84 2016-06-24     722       5081        552
    85 2016-06-25     907       5075        551
    86 2016-06-26    1234       5071        551
    87 2016-06-27    1490       5073        550
    88 2016-06-28    2346       5073        550
    89 2016-06-29    2398       5072        550
    
    [90 rows x 4 columns]
    

### Save_to_CSV:

As required in the problem statement, we are required to save the data in csv format containing the above columns. to_csv function will save the dataframe in the correct working directory.


```python
df.to_dense().to_csv("estimates.csv", index = False, sep=',', encoding='utf-8')
```

Let's find the top 10 app_id by sales by running a simple command as below:


```python
df1 = df.nlargest(10, 'sales_est', keep='first')
print(df1)
```

            Dates  app_id  sales_est  ranks_est
    5  2016-04-06    1490      13394        464
    4  2016-04-05    1234      11409        483
    10 2016-04-11     459      11334        465
    18 2016-04-19     406       8427        522
    7  2016-04-08    2891       8296        464
    13 2016-04-14    1490       8019        592
    26 2016-04-27     406       6849        542
    3  2016-04-04     722       6847        608
    20 2016-04-21     722       6536        570
    17 2016-04-18     320       6431        525
    

We can  visualize the app_id by total sales and take a quick look as to which app_id exhibited the most sales as per our forecast.


```python
df1.plot(x = 'app_id' , y = 'sales_est',kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15b0046c8d0>




![png](output_65_1.png)


We can also visualize top 10 ranks in terms of total sales.


```python
df1.plot(x = 'ranks_est' , y = 'sales_est',kind = 'barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15b0069e2b0>




![png](output_67_1.png)


### Summary & Insights :

The Main agenda was to find out an estimation of sales as a factor of ranks and see if there is any correlation between these two properties. We can surely derive that there is a correalation between these two as depicted in the graphs as well. SOme insights worth noting are as below:

1) The top 5 app_id by sales are 1490,1234,459,406, & 2891 as per our forecast model. This is also in alignment with the observed values from the given datasets.
2) There is surely an influence of higher ranks on higher sales. But, this cannot be drawn as an inference that the highest ranks will have the highest sales. Because, the app_id with the rank 483 has exhibited higher sales than 465 and so is the case with rank 570 and 525.

### Possible Enhancements:

Our script is only taking avg rank as an affecting factor on Sales as of now. If given more time on this, I think a possible improvement can be to feed the hourly rank data in our Model instead of avg rank so that we can bank upon the accuracy of the model. Also, for the sake of time complexity, we considered only two independent variables Date, Ranks but we can definitely consider ratings and other factors while forecasting the Sales values while expediting on the time complexity of the model too.


                                        
                                        ## Thank you ##
