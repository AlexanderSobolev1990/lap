# LAP - Linear Assignment Problem / Линейная дискретная оптимизационная задача (задача о назначениях) #

## 1. Brief / Обзор ##
<br/> Solving linear assignment problem using / Решение задачи о назначениях методами:
* Jonker-Volgenant-Castanon method (JVC) for dense and sparse (CSR - compressed sparse row) matrices / Метод Джонкера-Волгенанта-Кастаньона для плотных и разреженных матриц в CSR формате
* Mack method / Метод Мака
* Hungarian (Munkres) method / Венгерский алгоритм
* Murty K-best method (uses JVCsparse inside) / Метод Мурти k-лучших решений (используется алгоритм JVCsparse внутри)

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

<center><img src="doc/pictures/denseSmall.png" width="1000px"></center>
<?\image html  doc/pictures/denseSmall.png width=1000px?>
<?\image latex doc/pictures/denseSmall.png?> 
<center>Fig.1 - Execution time for dense matrices (small dimensions)<br /> </center>
<center>Рис.1 - Время выполнения на плотных матрицах (малые размернсти)<br /></center>

***

<center><img src="doc/pictures/sparseSmall.png" width="1000px" /></center>
<?\image html  doc/pictures/sparseSmall.png width=1000px?>
<?\image latex doc/pictures/sparseSmall.png?> 
<center>Fig.2 - Execution time for sparse matrices (small dimensions) <br /> </center>
<center>Рис.2 - Время выполнения на разреженных матрицах (малые размернсти)</center>


***

<center><img src="doc/pictures/denseLarge.png" width="1000px" /></center> 
<?\image html  doc/pictures/denseLarge.png width=1000?>
<?\image latex doc/pictures/denseLarge.png?>
<center>Fig.3 - Execution time for dense matrices (large dimensions) <br /> </center>
<center>Рис.3 - Время выполнения на плотных матрицах (большие размерности)</center>

***

<center><img src="doc/pictures/sparseLarge.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparseLarge.png width=1000?>
<?\image latex doc/pictures/sparseLarge.png?>
<center>Fig.4 - Execution time for sparse matrices (large dimensions) <br /> </center>
<center>Рис.4 - Время выполнения на разреженных матрицах (большие размерности)</center>

***

<center><img src="doc/pictures/sparseLargeExtra.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparseLargeExtra.png width=1000?>
<?\image latex doc/pictures/sparseLargeExtra.png?>
<center>Fig.5 - Execution time for sparse matrices (large dimensions) for JVCsparse<br /> </center>
<center>Рис.5 - Время выполнения на разреженных матрицах (большие размерности) для JVCsparse</center>

***

Additional graphs pics for methods of sequental extremum for dense and sparse matrices in dense and COOrdinated formats /
Дополнительные графики для методов последовательного выбора экстремума для плотных и разреженных матриц в обычном плотном виде и COO-формате.

***

<center><img src="doc/pictures/denseSmallSeqExtr.png" width="1000px" /></center> 
<?\image html  doc/pictures/denseSmallSeqExtr.png width=1000?>
<?\image latex doc/pictures/denseSmallSeqExtr.png?>
<center>Fig.6 - Execution time for dense matrices (small dimensions)<br /> </center>
<center>Рис.6 - Время выполнения на плотных матрицах (малые размерности)</center>

***

<center><img src="doc/pictures/sparseSmallSeqExtr.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparseSmallSeqExtr.png width=1000?>
<?\image latex doc/pictures/sparseSmallSeqExtr.png?>
<center>Fig.7 - Execution time for sparse matrices (small dimensions)<br /> </center>
<center>Рис.7 - Время выполнения на разреженных матрицах (малые размерности)</center>

## 5. Time measurements tables for sparse matrices / Сводные таблицы замеров времени выполнения для разреженных матриц

Table 1 - Execution time, milliseconds / Таблица 1 - Время выполнения, миллисекунды

<!--Таблица получена из теста time_table-->

| N | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hungarian | 0.007 | 0.046 | 0.369 | 4.358 | 66.083 | 1344.975 | - | - |
| Mack | 0.002 | 0.014 | 0.143 | 1.863 | 26.124 | 327.440 | - | - |
| SeqExtr | 0.004 | 0.018 | 0.141 | 1.336 | 16.591 | 238.068 | - | - |
| SeqExtrCOO | 0.002 | 0.009 | 0.045 | 0.297 | 2.663 | 39.466 | - | - |
| JVCdense | 0.002 | 0.005 | 0.020 | 0.083 | 0.418 | 3.185 | 26.182 | 586.341 |
| JVCsparse | 0.001 | 0.001 | 0.003 | 0.007 | 0.023 | 0.126 | 1.159 | 5.505 |

***

Table 2 - Increasing execution time relative to JVCsparse (times) / Таблица 2 - Возрастание времени выполнения относительно метода JVCsparse (разы)

<!--Таблица получена из теста time_table-->

| N | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hungarian | 6.371 | 32.886 | 124.864 | 653.129 | 2908.809 | 10701.411 | - | - |
| Mack | 2.162 | 10.120 | 48.250 | 279.142 | 1149.913 | 2605.306 | - | - |
| SeqExtr | 3.730 | 13.170 | 47.842 | 200.191 | 730.270 | 1894.205 | - | - |
| SeqExtrCOO | 2.141 | 6.424 | 15.241 | 44.439 | 117.197 | 314.015 | - | - |
| JVCdense | 1.601 | 3.438 | 6.687 | 12.485 | 18.380 | 25.343 | 22.596 | 106.506 |

## 6. Murty K-best method / Метод Мурти k-лучших решений

Assignment problem is solved in Murty's method calling JVCsparse algorithm inside. Method gives K-best assignment solutions using "warm" start with u- and v-variables in JVCsparse (warm start accelerates solution).
/
Внутри метода задача о назначениях решается путем вызова алгоритма JVCsparse. Метод дает K-лучших решений задачи о назначениях используя "теплый" старт при помощи u- и v-переменных в JVCsparse (теплый старт ускоряет метод).

Table 3 - Data for Murty's method / Таблица 3 - Исходная матрица для метода Murty

|  -  | j=0 | j=1 | j=2 | j=3 | j=4 | j=5 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| i=0 | 0.6 | 0.3 | 0.0 | 0.1 | 0.0 | 0.0 |
| i=1 | 0.0 | 0.2 | 0.7 | 0.0 | 0.1 | 0.0 |
| i=2 | 0.3 | 0.5 | 0.1 | 0.0 | 0.0 | 0.1 |

***

Table 4 - Solution for table 3 / Таблица 4 - Решение для таблицы 3

| K-best | Cost | Rowsol | Time, mcs |
| :-: | :-: | :-: | :-: |
| 1 | 1.8 | 0 2 1 | 56.646 |
| 2 | 1.4 | 0 2 5 | 6.133 |
| 3 | 1.3 | 1 2 0 | 22.13 |
| 4 | 1.2 | 0 4 1 | 13.752 |
| 5 | 1.1 | 1 2 5 | 5.063 |
| 6 | 0.8 | 0 4 2 | 3.911 |
| 7 | 0.7 | 1 4 0 | 11.031 |
| 8 | 0.5 | 1 4 2 | 1.7 |

## 7. Conclusion / Вывод

JVCsparse is the fastest method from considered (for sparse matrices), cause it works with compact CSR storage and uses fast JVC algorithm. JVCdense is the fastest for dense.
Method of sequental extremum is non-optimal and it's usage is not recommended.
/ 
JVCsparse самый быстрый метод среди рассмотренных (для разреженных матриц), поскольку работает с матрицами, хранящимися в компактном CSR формате и использует быстрый JVC алгоритм. Для плотных матриц самый быстрый JVCdense.
Метод последовательного выбора экстремума неоптимален и его использование не рекомендуется.
