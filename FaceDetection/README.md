Dans ce projet vous allez améliorer une implémentation existante de Viola-Jones qui s'exécute sur CPU. J'ai remplacé les parties du code qui font du calcul intensif par leurs version sur parallèle sur GPU. Le but est d'avoir une version plus rapide que la version séquentielle.


L'idée générale pour trouver un visage est de chercher des zones de dont le contraste est particulier. Par exemple, les yeux sont en général séparés par une partie plus sombre (le nez). Cette analyze est couteuse car elle doit être faite sur toutes les zones de l'image et à différentes échelles. Pour améliorer la performance, on a recours à du machine learning.


Le code utilise comme base durant ce projet : https://github.com/aparande/FaceDetection

Attention, il nécessite d'installer le framework Pickle avec PIP

