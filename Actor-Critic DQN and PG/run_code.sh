killall -v tensorboard
rm -rf ../tmp/*
clear
ipython -i ${1:-run_dqn_critic_cartpole.py}
