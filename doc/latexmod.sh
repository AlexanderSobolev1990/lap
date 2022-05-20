#!/bin/bash
filename=latex/refman.tex
sed -i '/{\\large Создано системой Doxygen /c \{\\large Программная документация}' $filename
sed -i 's|Создано системой Doxygen|Решение задачи о назначениях / Программная документация|' $filename
sed -i 's|\\setcounter{tocdepth}{3}|\\setcounter{tocdepth}{2}|' $filename
sed -i 's|\\begin{document}|\\begin{document}\n  \\renewcommand\\chaptername{Раздел}|' $filename
cd latex && make 
wait
cd ../
cp -R latex/refman.pdf ./Задача_о_назначениях_Программная_документация.pdf
