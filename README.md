# MTGradient

This is an attempt to learn to draft using the data provided by [17Lands](https://www.17lands.com/public_datasets). 

### Background

The end product of this repo is a Dash app that parses your MTG Arena log (hopefully other sources later) and provides statistics and model recommendations during a draft. The code for parsing the log comes from 
the [17Lands parser](https://github.com/rconroy293/mtga-log-client). Articles and code from [Ryan Saxe](https://github.com/RyanSaxe) provided benchmarks and helpful insights on encouraging the model to learn 
good habits.

### Model

The model is a simple transformer with access to three sources of information:

- The current pick options
- The cards you've already picked
- The whole history of the cards you've seen

Each card is passed through an embedding layer (randomly initialized for now), the history is aggregated per-round to `(batch_size, n_rounds, emb_dim)`, then the three sources are concatenated together.

The transformer is required to learn two tasks: 

- Predict the card the user chose in the given pack
- Predict the number of wins the deck will get given the current pool

### Data

To help the model learn to draft well, a good chunk of the data is either downweighted or ignored. All Brozne and Silver drafts are dropped, and the loss on each example is weighted by the rank, deck win rate, and user win rate. The win prediction is not weighted by win rates but is weighted by rank and the round (since you should know a lot more about the deck in pack 3 than pack 1 pick 1). 

Each round gets a different loss weighting to reflect both its importance and its predictability. The first pick is not heavily weighted since it often contains rare-drafting. The next several picks are all weighted equally, with importance dropping of quickly - the final few decisions contribute very little to the loss.  

Any drafts that seem incomplete (0 wins, 0 losses) are dropped. Any incomplete drafts (fewer than 7 wins but fewer than 3 losses) are weighted 0 for win prediction.

The dataset is split by time: the final period of the draft is the validation set and the rest is the training set. Hopefully this makes the validation set approximately mirror the way a competition would work if it cycled back around on Arena today.

Note: Scryfall provides card data like `cmc` which is useful for initializing the card representations.

### Results

Using the NEO Premier Draft data and preprocessing as above, an early model without much hyperparameter tuning achieves about 71% validation accuracy, where the validation set is just the final day of drafting (1260 drafts).

One thing to keep an eye on is using the bot in Quick Draft seems to result in some surprising behavior. The model appears to pick up on any indication that a color is "open" (despite it really being a difference between how humans draft and the Arena bot drafts), which in an early iteration of the model led to a fairly consistent swing towards W in Quick Draft.

### UI

The Dash app monitors and parses the Arena log then displays both the 17Lands card rating data and the model's predictions together. It also guesses which lanes may be open (by tracking `seen_at - alsa` per color), and gives an overview of your pool so far.