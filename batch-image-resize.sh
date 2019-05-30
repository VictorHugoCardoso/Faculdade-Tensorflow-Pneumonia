SIZE=256
FOLDER1="/home/nit/Documentos/tflearn/v1-pneumonia/jpg/0"
FOLDER2="/home/nit/Documentos/tflearn/v1-pneumonia/jpg/1"
find ${FOLDER1} -iname '*.jpeg' -exec convert \{} -verbose -resize ${SIZE}x${SIZE}! \{} \;
find ${FOLDER2} -iname '*.jpeg' -exec convert \{} -verbose -resize ${SIZE}x${SIZE}! \{} \;
