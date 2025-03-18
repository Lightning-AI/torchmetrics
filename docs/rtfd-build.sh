# building for PRs and skip stable and latest states

export SPHINX_ENABLE_GALLERY=0

if ! [ $READTHEDOCS_VERSION == "latest" -o $READTHEDOCS_VERSION == "stable" ];
then
    cd ./docs ;
    export SPHINX_FETCH_ASSETS=0 ;
    make html --jobs $(nproc) ;
    ls -lh build
else
    echo "Void build... :-]" ;
    mkdir -p ./docs/build/html
    cp ./docs/redirect.html ./docs/build/html/index.html
fi
