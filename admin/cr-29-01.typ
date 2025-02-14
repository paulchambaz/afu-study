#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [Compte Rendu de Réunion 2],
  course: [Projet #smallcaps[Ai2d]],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [Sorbonne Université],
  reference: [Master #smallcaps[Ai2d] M1 -- 29 janvier 2025],
)

== Points clés

- Commencer par l'implémentation PyTorch d'AFU avant l'intégration BBRL.
- Structurer le développement des algorithmes progressivement : Q-learning $->$ DQN $->$ DDPG $->$ SAC.
- Utiliser CartPole continuous comme environnement de test.
- Il faut implémenter _set_state_ pour générer des états uniformes.
- Discussion sur les wrappers de gymnasium.
- Points de cours sur la progression DQN → DDPG → SAC : DQN adapte Q-learning aux espaces d'états continus via les réseaux de neurones, DDPG étend cela aux actions continues avec une architecture acteur-critique, et SAC améliore DDPG en ajoutant de l'entropie pour l'exploration et des Q-functions multiples pour la stabilité.

== Pour la semaine prochaine

=== AFU
- Récupérer et tester le code PyTorch AFU de M. Perrin

=== CartPole
- Intégrer bbrl_gymnasium avec CartPole continuous.
- Implétmenter _set_state_ dans l'environment.

=== Algorithmes
- Implémenter DQN.
- Implémenter DDPG.
- (Si possible) Implémenter SAC.
- Tester chaque algorithme avec CartPole continuous.
