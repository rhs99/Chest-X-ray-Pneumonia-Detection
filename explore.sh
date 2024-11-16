echo 'train----------'
echo 'normal'
ls chest_xray/train/NORMAL/ | wc -l
echo 'bacteria'
ls chest_xray/train/PNEUMONIA/ | grep bacteria | wc -l
echo 'virus'
ls chest_xray/train/PNEUMONIA/ | grep virus | wc -l

echo 'val----------'
echo 'normal'
ls chest_xray/val/NORMAL/ | wc -l
echo 'bacteria'
ls chest_xray/val/PNEUMONIA/ | grep bacteria | wc -l
echo 'virus'
ls chest_xray/val/PNEUMONIA/ | grep virus | wc -l

echo 'test----------'
echo 'normal'
ls chest_xray/test/NORMAL/ | wc -l
echo 'bacteria'
ls chest_xray/test/PNEUMONIA/ | grep bacteria | wc -l
echo 'virus'
ls chest_xray/test/PNEUMONIA/ | grep virus | wc -l