#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 7],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 26 mars 2025],
)

== Points clés

- Explication sur nos résultats sur Lunar Lander, Pendulum et Cartpole en Offpolicy et Onpolicy et tout le code que l'on a refait avec BBRL pour les Q, V, A et Policy networks qui utilisent un agent BBRL.
- Mise au point sur Cal-QL notamment sur le monte carlo return, l'usage du replay buffer dans l'expérience Offline to Online et le graphe qui présente l'expérience qui n'est pas bien expliqué par ses auteurs.
- Mise au point sur le Offline to Online, à voir si l'on continue dessus après la semaine prochaine ou si l'on se focalise sur le Offpolicy et le Onpolicy.

== Axes de travail

- Continuer les expériences sur la partie Offpolicy et Onpolicy, en particulier sur Lunar Lander.
- Finir l'implémentation de Cal-QL et de IQL.
- Continuer la partie Offline to Online, garder le replay buffer comme dans Cal-QL, faire tourner les premières expériences de transition.
