config = {
    "BATCH_SIZE": 128,  # minibatch size
    "MEMORY_CAPACITY": 7000,  # replay memory
    "EPISODES_PER_EPOCH": 100,  # for training
    "NUMBER_OF_EPOCHS": 5,  # for training
    "FRAME_SKIP": 10,  # number of frames to skip per action
    "FRAME_STACK": 3,  # number of frames to stack
    "LEARNING_RATE": 1e-3,  # alpha learning
    "WEIGHT_DECAY": 1e-5,  # optimizer weight decay
}
