# **OnMyFeed-ML-**  
<p style="text-align: center;">This repository contains the ml part of the project. It is currently in the alpha version, so functionality and performance may change significantly as development progresses.
  
An application with a cross-platform intelligent recommendation system based on machine learning that analyzes content from different social networks and provides personalized recommendations to marketers. It is scalable, fault-tolerant, and supports automatic deployment and monitoring.


---

### **Advantages**
- Scalability: allows you to add additional networks to expand your recommendations
- Flexibility: there's a possibility of sensitive adjustment and combination with other models (example: Caser or Bert4Rec)
- Powerful Pipeline : the solution uses an ensemble of transformers, which allows you to give the most accurate recommendations
- Expansion: we are aware of the frequent problems that exist in RecSys tasks, so we wrote the modular project code in advance so that we could later plug the .enhance module


---

### **Technologies**
- **Schedule**: for planning tasks
- **Selenium**: for data collection
- **Optuna**: for tuning hyperparameters
- **CatBoost**: prediction and ranks the most popular and similar content
- **PyTorch**: reduces the search space to the most relevant options using embedding in an ensemble model based on transformers
