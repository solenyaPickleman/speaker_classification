for f in $(ls *.flac) 
do
	sox $f ${f/flac/wav}
done

