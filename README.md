# LAP - Linear Assignment Problem / Линейная дискретная оптимизационная задача (задача о назначениях) #

## 1. Brief / Обзор ##
<br/> Solving linear assignment problem using / Решение задачи о назначениях методами:
* Jonker-Volgenant-Castanon method (JVC) for dense and sparse (CSR - compressed sparse row) matrices / Метод Джонкера-Волгенанта-Кастаньона для плотных и разреженных матриц в CSR формате
* Mack method / Метод Мака
* Hungarian (Munkres) method / Венгерский алгоритм

## 2. References / Ссылки ##
Papers / Статьи:
* R.Jonker and A.Volgenant A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems Computing 38, 325-340 (1987)
* A.Volgenant Linear and Semi-Assignment Problems: A Core Oriented Approach
* Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123

Sites / Сайты:
* http://www.assignmentproblems.com/linearAP.htm
* https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0

Repositories / Репозитории:
* https://github.com/yongyanghz/LAPJV-algorithm-c
* https://github.com/RcppCore/rcpp-gallery/blob/gh-pages/src/2013-09-24-minimal-assignment.cpp
* https://github.com/fuglede/linearassignment

## 3. Dependencies / Зависимости ##
<br /> Armadillo for matrices, Boost for testing / Armadillo для работы с матрицами, Boost для тестирования.

## 4. Tests / Тесты ##
* Сomparison of calculation speed on dense and sparse matrices / Сравнение скорости работы на плотных и разреженных матрицах
* Simple assignment problem matrices are provided / Дополнительные тесты на простых матрицах
* test JVC algorithm for looping / Тест алгоритма JVCdense на зацикливание

(Sparsity is ~20% / В разреженной матрице ~20% назначенных ячеек)

***
<center>Results for time measuring / Результаты замеров скорости работы:</center>

<center><img src="doc/pictures/dense5to50.png" width="1000px"></center>
<?\image html  doc/pictures/dense5to50.png width=1000px?>
<?\image latex doc/pictures/dense5to50.png?> 
<center>Fig.1 - Execution time for dense matrices (small dimensions)<br /> </center>
<center>Рис.1 - Время выполнения на плотных матрицах (малые размернсти)<br /></center>

***

<center><img src="doc/pictures/dense100to1000.png" width="1000px" /></center> 
<?\image html  doc/pictures/dense100to1000.png width=1000?>
<?\image latex doc/pictures/dense100to1000.png?>
<center>Fig.2 - Execution time for dense matrices (large dimensions) <br /> </center>
<center>Рис.2 - Время выполнения на плотных матрицах (большие размерности)</center>

***

<center><img src="doc/pictures/sparse5to50.png" width="1000px" /></center>
<?\image html  doc/pictures/sparse5to50.png width=1000px?>
<?\image latex doc/pictures/sparse5to50.png?> 
<center>Fig.3 - Execution time for sparse matrices (small dimensions) <br /> </center>
<center>Рис.3 - Время выполнения на разреженных матрицах (малые размернсти)</center>

***

<center><img src="doc/pictures/sparse100to1000.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparse100to1000.png width=1000?>
<?\image latex doc/pictures/sparse100to1000.png?>
<center>Fig.4 - Execution time for sparse matrices (large dimensions) <br /> </center>
<center>Рис.4 - Время выполнения на разреженных матрицах (большие размерности)</center>

***

<center><img src="doc/pictures/sparse500to5000.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparse500to5000.png width=1000?>
<?\image latex doc/pictures/sparse500to5000.png?>
<center>Fig.5 - Execution time for sparse matrices (large dimensions) for JVCsparse<br /> </center>
<center>Рис.5 - Время выполнения на разреженных матрицах (большие размерности) для JVCsparse</center>

***

Additional graphs pics for methods of sequental extremum for dense and sparse matrices in dense and COOrdinated formats /
Дополнительные графики для методов последовательного выбора экстремума для плотных и разреженных матриц в обычном плотном виде и COO-формате.

***

<center><img src="doc/pictures/dense5to50SeqExtr.png" width="1000px" /></center> 
<?\image html  doc/pictures/dense5to50SeqExtr.png width=1000?>
<?\image latex doc/pictures/dense5to50SeqExtr.png?>
<center>Fig.6 - Execution time for dense matrices (small dimensions)<br /> </center>
<center>Рис.6 - Время выполнения на плотных матрицах (малые размерности)</center>

***

<center><img src="doc/pictures/sparse5to50SeqExtr.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparse5to50SeqExtr.png width=1000?>
<?\image latex doc/pictures/sparse5to50SeqExtr.png?>
<center>Fig.7 - Execution time for sparse matrices (small dimensions)<br /> </center>
<center>Рис.7 - Время выполнения на разреженных матрицах (малые размерности)</center>

## 5. Time measurements tables for sparse matrices / Сводные таблицы замеров времени выполнения для разреженных матриц

Table 1 - Execution time, milliseconds / Таблица 1 - Время выполнения, миллисекунды

<!--Таблица получена из теста time_table-->

| N | 5 | 10 | 25 | 50 | 100 | 150 | 200 | 250 | 500 | 1000 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hungarian | 0.013 | 0.095 | 1.931 | 24.129 | 253.358 | 1772.925 | 6687.237 | - | - | - |
| Mack | 0.005 | 0.037 | 0.801 | 10.185 | 135.415 | 627.683 | 1815.623 | - | - | - |
| SeqExtr | 0.006 | 0.035 | 0.577 | 6.556 | 88.152 | 433.079 | 1343.761 | - | - | - |
| SeqExtrCOO | 0.003 | 0.015 | 0.152 | 1.216 | 15.856 | 83.929 | 323.072 | - | - | - |
| JVCdense | 0.003 | 0.009 | 0.052 | 0.220 | 1.400 | 4.562 | 10.246 | 20.598 | 232.334 | 3876.377 |
| JVCsparse | 0.001 | 0.002 | 0.005 | 0.013 | 0.049 | 0.132 | 0.261 | 0.535 | 4.481 | 21.902 |

***

Table 2 - Increasing execution time relative to JVCsparse (times) / Таблица 2 - Возрастание времени выполнения относительно метода JVCsparse (разы)

<!--Таблица получена из теста time_table-->

| N | 5 | 10 | 25 | 50 | 100 | 150 | 200 | 250 | 500 | 1000 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hungarian | 10.914 | 46.775 | 407.213 | 1884.278 | 5146.364 | 13436.377 | 25614.081 | - | - | - |
| Mack | 4.437 | 17.944 | 168.873 | 795.364 | 2750.645 | 4756.991 | 6954.369 | - | - | - |
| SeqExtr | 4.789 | 17.179 | 121.615 | 511.956 | 1790.602 | 3282.153 | 5147.001 | - | - | - |
| SeqExtrCOO | 2.848 | 7.351 | 32.091 | 94.928 | 322.073 | 636.071 | 1237.459 | - | - | - |
| JVCdense | 2.246 | 4.249 | 10.926 | 17.158 | 28.438 | 34.577 | 39.247 | 38.476 | 51.850 | 176.988 |

## 6. Conclusion / Вывод

JVCsparse is the fastest method from considered (for sparse matrices), cause it works with compact CSR storage and uses fast JVC algorithm. JVCdense is the fastest for dense.
Method of sequental extremum is non-optimal and it's usage is not recommended.
/ 
JVCsparse самый быстрый метод среди рассмотренных (для разреженных матриц), поскольку работает с матрицами, хранящимися в компактном CSR формате и использует быстрый JVC алгоритм. Для плотных матриц самый быстрый JVCdense.
Метод последовательного выбора экстремума неоптимален и его использование не рекомендуется.
