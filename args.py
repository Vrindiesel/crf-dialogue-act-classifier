

glove_path = "/home/diesel/Projects/Datasets/Datasets/glove_data/glove.840B/glove.840B.300d.txt"

def add_args(parser):
    group = parser.add_argument_group('General')
    #group.add_argument('-save_model', #required=True,
    #                   help="Output path for model checkpoints")
    group.add_argument('-seed', type=int, default=2323,
                       help="""Random seed used for the experiments
                       reproducibility.""")
    group.add_argument('-save_model', default='model',
                       help="""Model filename (the model will be saved as """)

    group.add_argument('-cuda', action="store_true",
                       help="""Use GPU device 0.""")

    group.add_argument('-batch_size', type=int, default=1,
                       help='Number of conversations in a batch.')
    group.add_argument('-epochs', type=int, default=13,
                       help='Number of training epochs')
    group.add_argument('-optim', default='sgd',
                       choices=['sgd', 'adagrad',  'adam', "adadelta"],
                       help="""Optimization method.""")
    group.add_argument('-max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")

    group.add_argument('-dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")

    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=1.0,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_decay', type=float, default=0.5,
                       help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past
                       start_decay_at""")
    group.add_argument('-start_decay_at', type=int, default=8,
                       help="""Start decaying every epoch after and including this
                       epoch""")
    group.add_argument('-start_checkpoint_at', type=int, default=0,
                       help="""Start checkpointing every epoch after and including
                       this epoch""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=50,
                       help="Print stats at this interval.")

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-word_vec_size', type=int, default=300,
                       help='Word embedding size.')
    group.add_argument('-fix_word_vecs',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")


    # Model Options
    group = parser.add_argument_group('Model')
    group.add_argument('-layers', type=int, default=1,
                       help='Number of rnn layers')
    group.add_argument('-rnn_size', type=int, default=600,
                       help='Size of rnn hidden states')

    group.add_argument('-load_model', default=None, help="Path to the model checkpoint.")

    # Data Options
    group = parser.add_argument_group('Data')
    group.add_argument('-data_path', #required=True,
                       help="Path to the corpus swda-corpus.json")
    group.add_argument('-glove_path',
                       help="Path to the corpus swda-corpus.json")
    group.add_argument('-save_data', #required=True,
                       help="Output path for the prepared data")
    group.add_argument('-word_min_frequency', type=int, default=0)
    group.add_argument('-load_data', action="store_true",
                       help="""If dataset already exists in save_dir then those
                       files will be loaded instead of creating a new ones.""")
    group.add_argument('-lower', action="store_true",
                       help="""Lower case the input data.""")
    group.add_argument('-simulate_data', action="store_true",
                       help="""use simulated data during inference.""")

