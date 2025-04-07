#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [Ordre du jour],
  course: [Projet #smallcaps[Ai2d]],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [Sorbonne Université],
  reference: [Master #smallcaps[Ai2d] M1 -- 26 mars 2025],
)

== Travail accompli

- Réécriture des algorithmes utilisés AFU, SAC et DDPG pour mieux
comprendre et intégrer BBRL.

- Beaucoup d'expériences:

     On-policy: AFU et SAC sur Cartpole, Pendulum et Lunar Lander

     Off-policy: AFU et SAC sur Cartpole, Pendulum et Lunar Lander

     Random walk policy: AFU et SAC sur Pendulum et Lunar Lander

     Hybrid policy: AFU et SAC sur Lunar Lander

     Optimal loss off-policy: AFU et SAC sur Pendulum et Lunar Lander

- Résolution des problèmes Mujoco: on peut maintenant tester les sept
environnements du papier original, nous sommes passés à uv pour gérer
les dépendances de façon plus "python"

- Ecriture de trois nouvelles expériences:

     - Random walk policy commence un épisode à un état choisi
uniformément dans l'espace d'état puis réalise des actions choisies
uniformément dans l'espace d'action jusqu'à la fin de l'épisode. Cette
expérience permet de voir si de simples actions aléatoires permettent de
converger.

     - Hybrid policy utilise epsilon (in [0, 1]) pour apprendre soit
on-policy soit off-policy (compte-tenu d'un tirage aléatoire à chaque
pas). Cette expérience permet de voir l'impact de donner off-policy dans
le replay buffer sur la dégradation des performances de l'algorithme. On
teste epsilon in {0.0, 0.1, 0.2, ..., 1.0}.

     - Optimal loss off-policy utilise une politique optimale
pré-entrainée sur les loss et un entrainement off-policy classique.
Cette expérience permet de voir si l'apprentissage off-policy ne
converge pas simplement à cause d'un soucis de propagation de Q-valeurs.

- Ecriture d'un script de démo pour visualiser les poids d'un algorithme
sur une expérience donnée, ce qui permet de rejouer des expériences.
Très simple et nous aurions dû écrire ce script il y a longtemps car
c'est très utile pour mieux comprendre les environnements et les
résultats (surtout les cas d'échecs et les cas très proches de
l'optimum) des expériences.

- Intégration de checkpoints pour reprendre les expériences. De même,
cette fonctionnalité aurait due être écrite plus tôt.

- Travail sur une pipeline Grafana + InfluxDB + Jenkins pour lancer et
visualiser des expériences plus rapidement et facilement.

- Nous avons réalisé la présence de petits bugs durant le off-poolicy
(espace d'état divisé par deux, actions toujours prises entre -1 et 1 au
lieu de scaled). En les corrigeant, AFU et SAC convergent sur Pendulum
(ci-joint les graphes d'évolution) ! On a aussi pu faire tourner AFU et
SAC sur Lunar Lander et aucun des deux ne converge (ci-joint les graphes
d'évolution). Nous avons aussi pu tester la random walk policy qui n'est
pas très intéressante, SAC converge sur Pendulum, AFU ne converge pas
sur Pendulum, SAC et AFU ne convergent pas sur Lunar Lander.
L'expérience hybrid policy tourne actuellement et optimal loss
off-policy tournera d'ici à demain.

== Points à discuter

- Revenir sur les différentes expériences et sur l'hypothèse évoquée sur
mattermost.

- Revenir sur certains aspects de BBRL encore mal compris.

- Si le temps nous le permet, nous aimerions avoir des indications sur
l'apprentissage offline (CalQL, IQL) pour mieux comprendre l'intuition
derrière notre hypothèse.

- Retour sur la quantité de travail fournie sur le projet par les étudiants.