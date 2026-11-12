import math
import argparse
import os

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", default='../../Recdap/dataset/FB15K', type=str)
	parser.add_argument("--device", default='cuda:0', type=str)
	parser.add_argument("--random_embed", default=False, type=str,
						help='using pre-trained embeddings or initiaizing embeddings')
	parser.add_argument("--few", default=5, type=int)
	parser.add_argument("--test", action='store_true')
	parser.add_argument("--embed_model", default='TransE', type=str)
	parser.add_argument("--max_neighbor", default=100, type=int)
	parser.add_argument("--hidden_size", default=100, type=int)

	parser.add_argument("-epo", "--epoch", default=100000, type=int)

	parser.add_argument("--n_layers_tf", default=4, type=int)
	parser.add_argument("--num_head", default=4, type=int)
	parser.add_argument("--dropout_input", default=0.2, type=float)
	parser.add_argument("--dropout_TF", default=0.3, type=float)

	parser.add_argument("--num_query", default=128, type=int)

	parser.add_argument("--batch_size", default=512, type=int)
	parser.add_argument("--use_pretrain", default=True, type=str)

	parser.add_argument("--fine_tune", action='store_true')

	parser.add_argument('--load_model', default=False, type=str, help='Load existing model?')
	parser.add_argument("--eval_every", default=10000, type=int)

	parser.add_argument("--train_few", default=1, type=int)
	parser.add_argument("--prefix", default='intial', type=str)
	parser.add_argument("--random_seed", default=1, type=int)

	parser.add_argument("--seed", default='19950902', type=int)
	parser.add_argument("--max_batches", default=300000, type=int)
	parser.add_argument("--weight_decay", default=0, type=float)

	parser.add_argument("--lr", default=5e-5, type=float)

	parser.add_argument("--symbol_embed_lr", default=5e-5, type=float) #fix
	parser.add_argument("--base_lr", default=5e-4, type=float)  #fix

	parser.add_argument("--warm_up_step", default=10000, type=int)
	parser.add_argument("--optimizer", type=str, default="Adam",
						help="Which optimizer to use?")
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--margin", type=float, default=5.0,
						help="The margin between positive and negative samples in the max-margin loss") #original 5.0
	parser.add_argument("--grad_clip", default=5.0, type=float)
	parser.add_argument("--log_every", default=50, type=int)

	parser.add_argument("--support_aug", choices=("none", "gaussian", "mixup"), default="none")
	parser.add_argument("--aug_per_support", type=int, default=20)
	parser.add_argument("--aug_sigma", type=float, default=0.01)
	parser.add_argument("--checkpoint", help="Full CIAN state_dict file")
	parser.add_argument("--eval_split", choices=("test", "dev"), default="test")
	parser.add_argument("--candidate_chunk", type=int, default=256)
	parser.add_argument("--train_cgnet", action="store_true")
	parser.add_argument("--cg_checkpoint", help="Output for CGNet training; input for CGNet evaluation")
	parser.add_argument("--cg_variant", choices=("s", "r"), default="r")
	parser.add_argument("--cg_top_m", type=int, default=500)
	parser.add_argument("--cg_min_pairs", type=int, default=10)
	parser.add_argument("--cg_hidden", type=int, default=256)
	parser.add_argument("--cg_steps", type=int, default=100000)
	parser.add_argument("--cg_eval_every", type=int, default=5000)
	parser.add_argument("--cg_critic_steps", type=int, default=5)
	parser.add_argument("--cg_batch_size", type=int, default=64)
	parser.add_argument("--cg_lr", type=float, default=1e-5)
	parser.add_argument("--cg_beta", type=float, default=10.0)
	parser.add_argument("--cg_margin", type=float, default=5.0)
	parser.add_argument("--cg_rank_weight", type=float, default=1.0)
	parser.add_argument("--cg_noise_variance", type=float, default=1.0)
	parser.add_argument("--cg_draws", type=int, default=20)
	parser.add_argument("--cg_temperature", type=float, default=1.0)
	args = parser.parse_args()
	if args.train_cgnet and (args.test or not args.cg_checkpoint):
		parser.error("--train_cgnet needs --cg_checkpoint and cannot be combined with --test")
	if args.cg_checkpoint and (not args.checkpoint or args.support_aug != "none" or
	                           not (args.train_cgnet or args.test)):
		parser.error("CGNet requires --checkpoint and its own training or test mode")
	if args.cg_checkpoint and os.path.realpath(args.cg_checkpoint) == os.path.realpath(args.checkpoint):
		parser.error("Use distinct files for CIAN and CGNet checkpoints")
	if min(args.few, args.cg_top_m, args.cg_min_pairs, args.cg_hidden, args.cg_steps,
	       args.cg_eval_every, args.cg_critic_steps, args.cg_batch_size, args.cg_draws) < 1:
		parser.error("Shot count and CGNet integer arguments must be positive")
	if any(not math.isfinite(value) or value <= 0 for value in
	       (args.cg_lr, args.cg_margin, args.cg_noise_variance, args.cg_temperature)):
		parser.error("Invalid positive CGNet hyperparameter")
	if any(not math.isfinite(value) or value < 0 for value in (args.cg_beta, args.cg_rank_weight)):
		parser.error("CGNet loss weights must be finite and nonnegative")

	if args.support_aug != "none" and not args.test:
		parser.error("Use --test with a frozen CIAN checkpoint for augmentation")
	if args.aug_per_support < 1 or args.candidate_chunk < 1 or not math.isfinite(args.aug_sigma) or args.aug_sigma < 0:
		parser.error("Invalid augmentation count, noise scale or candidate chunk size")
	if args.support_aug == "mixup" and args.few < 2:
		parser.error("Mixup needs at least two supports")

	if not os.path.exists('models'):
		os.mkdir('models')
	args.save_path = 'models/' + args.prefix


	return args

