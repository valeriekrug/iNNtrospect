for i in `seq 0 1 7`
do
  layerid=$(printf "%03d" $i)
  python3 -u experiments/jair/03_classifier_probes.py VGG16 layer$layerid
done
for i in `seq 0 1 19`
do
  layerid=$(printf "%03d" $i)
  python3 -u experiments/jair/03_classifier_probes.py ResNet50 layer$layerid
done
for i in `seq 0 1 14`
do
  layerid=$(printf "%03d" $i)
  python3 -u experiments/jair/03_classifier_probes.py InceptionV3 layer$layerid
done
