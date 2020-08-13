# Differences Barplot

## What was the goal?
We want to know whether removing the smells that repeat themselves would improve the performance of our models.

In the end we tested the following configurations:
* Designite
* Designite + Fowler
* (Designite + Fowler) - Designite(Fowler)
* Designite - Designite(Fowler)
* Fowler

We proved that even though they are included in the same set, they are not the same.
For instance, the Multifaceted abstraction has the large class as an alias. We can see that by removing the Multifaceted Abstraction, there is a difference in the model's performances.

We can say the Designite is a generalization from the fowler.