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

<center><img src="doc/pictures/dense50to1000.png" width="1000px" /></center> 
<?\image html  doc/pictures/dense50to1000.png width=1000?>
<?\image latex doc/pictures/dense50to1000.png?>
<center>Fig.2 - Execution time for dense matrices (large dimensions) <br /> </center>
<center>Рис.2 - Время выполнения на плотных матрицах (большие размерности)</center>

***

<center><img src="doc/pictures/sparse5to50.png" width="1000px" /></center>
<?\image html  doc/pictures/sparse5to50.png width=1000px?>
<?\image latex doc/pictures/sparse5to50.png?> 
<center>Fig.3 - Execution time for sparse matrices (small dimensions) <br /> </center>
<center>Рис.3 - Время выполнения на разреженных матрицах (малые размернсти)</center>

***

<center><img src="doc/pictures/sparse50to1000.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparse50to1000.png width=1000?>
<?\image latex doc/pictures/sparse50to1000.png?>
<center>Fig.4 - Execution time for sparse matrices (large dimensions) <br /> </center>
<center>Рис.4 - Время выполнения на разреженных матрицах (большие размерности)</center>

***

<center><img src="doc/pictures/sparse500to7000.png" width="1000px" /></center> 
<?\image html  doc/pictures/sparse500to7000.png width=1000?>
<?\image latex doc/pictures/sparse500to7000.png?>
<center>Fig.5 - Execution time for sparse matrices (large dimensions) for JVCsparse<br /> </center>
<center>Рис.5 - Время выполнения на разреженных матрицах (большие размерности) для JVCsparse</center>

## 5. Time measurements tables / Сводные таблицы замеров времени выполнения

Table 1 - Execution time, milliseconds / Таблица 1 - Время выполнения, миллисекунды

| N | 5 | 10 | 25 | 50 | 100 | 150 | 200 | 250 | 500 | 1000 |	
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hungarian | 0.016 | 0.086 | 1.680 | 21.895 | 349.339 | 1811.32 | 6618.14 | - | - | - |	
| Mack      | 0.005 | 0.035 | 0.713 | 9.673 | 125.7 | 577.939 | 1703.67 | - | - | - |	
| JVCdense  | 0.003 | 0.008 | 0.052 | 0.226 | 1.421 | 4.511 | 10.346 | 20.888 | 232.497 | 3921.59 |	
| JVCsparse | 0.001 | 0.001 | 0.004 | 0.013 | 0.050 | 0.136 | 0.271 | 0.568 | 4.623 | 22.131 |

***

Table 2 - Increasing execution time relative to JVCsparse (times) / Таблица 2 - Возрастание времени выполнения относительно метода JVCsparse (разы)

| N | 5 | 10 | 25 | 50 | 100 | 150 | 200 | 250 | 500 | 1000 |	
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hungarian | 15.434 | 45.964 | 353.878 | 1589.14 | 6935.4 | 13247 | 24384.1 | - | - | - |	
| Mack      | 5.161 | 18.643 | 150.171 | 702.077 | 2495.52 | 4226.73 | 6277.05 | - | - | - |	
| JVCdense  | 3.129 | 4.480 | 11.137 | 16.455 | 28.214 | 32.996 | 38.1224 | 36.722 | 50.285 | 177.192 |	

## 6. Conclusion / Вывод

JVCsparse is the fastest method from considered / JVCsparse самый быстрый метод среди рассмотренных
