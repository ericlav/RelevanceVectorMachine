# Régression Bayésienne avec les Relevance Vector Machine

Ceci est le répertoire contenant les fichiers de code pour le projet de Christos Katsoulakis et Eric Lavergne.

Le script *RVM.py* contient les différentes fonctions que nous avons implémentées pour mettre en oeuvre les Relevance Vector Machine (RVM), il peut être lu avec un éditeur de texte. Nous y définissions tout d'abord différents kernels pouvant être utilisés : linear kernel, linear spline kernel, gaussian kernel et rbf kernel. Ensuite, pour définir le modèle Relevance Vector Machine nous avons adopté les pratiques de scikit-learn pour avoir un modèle facilement utilisable dans d'autres contextes. C'est ainsi que nous avons codé une classe RVM qui contient comme fonctions principales : \_\_init\_\_ (initialisation des paramètres généraux de la classe), fit_init (initialisation des paramètres du modèle), fit (apprentissage du modèle), predict (prédictions ponctuelles du modèle) et predict_proba (prédictions probabiliste du modèle). Les autres fonctions servent de support à ces principales fonctions.

Le notebook *Demo.ipynb* contient les différentes expériences que nous avons réalisées qui ont permis d'aboutir aux résultats  présentés dans la Section 4 de notre rapport, il peut être lu avec l'application Jupyter Notebook.