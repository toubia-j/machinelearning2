# machinelearning2
Si vous désirez exécuter le projet localement, voici la procédure :

Téléchargez le contenu du FileSender (les modèles) : https://filesender.renater.fr/?s=download&token=528c7b46-0ee4-40b4-a0fc-89ff81e7fecc

Dans un premier terminal, placez-vous à la racine du projet et lancez :
-python .\serving\test.py

-Dans un deuxième terminal, toujours à la racine du projet, lancez :

-python .\serving\app.py

-Chaque script lancera un serveur Flask sur des ports distincts. Vous pourrez ensuite interagir avec le projet et tester les différents modèles.

-Ensuite le page web sera disponible en local http://127.0.0.1:5001
