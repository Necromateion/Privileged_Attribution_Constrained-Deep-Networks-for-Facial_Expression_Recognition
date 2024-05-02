# DeepNN
Martin Jarnier
Bryan Chen
Andy Kiuchi
Imad El-Mansouri

pensez a creer un dossier misc avec le dataset que vous pouvez trouver ici https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/data.
Le papier quand a lui est ici -> https://arxiv.org/abs/2203.12905
Le but c'est d'utiliser des heatmap qui focus l'attention du modele sur des points precis du visages. ON utilise les heatmaps quand on calcul la loss dans la boucle d'entrainement on va venir les comparer a des carte d'attribution qu'on genere avec differentes methodes: ici on en a deux soit gracam soit grad_paper qui est l'implementation directement issue du papier. ON a aussi un custom dataset qui permet d'avoir un dataloader qui prend a chaque fois l'image et sa heatmpas associer dans le dossier misc/heatmaps.
