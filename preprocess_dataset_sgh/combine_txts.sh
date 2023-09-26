if [ "$1" = "--path" ]; then
    cd "$2"
else
    echo "wrong usage, pls use . combine_txts.sh --path path/to/sgh_dir/"
fi
sed -i -e '$a\' *txt 
cat *txt > ../all.txt