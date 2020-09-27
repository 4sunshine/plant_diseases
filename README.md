# plant_diseases

**STRUCTURE**:  
*model.py* contains basic models: backbone(feature extractor) and full model.  
Full model HealthyPlant implements general end-to-end pipeline for plant disease 
classification.  
*dataset.py* dataset ops.  
*evaluate.py* script for getting metrics from all classifiers and for saving
best trained sklearn classifier.  
*features.py* get & save features.  
*inference.py* script for single image prediction of full model (E2E).  
*single_clf_cross_val.py* cross val example for single classifier.  
*utils.py* miscellaneous functions used in other scripts.  
    
