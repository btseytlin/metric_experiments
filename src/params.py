batch_size = 32
num_epochs = 40

test_interval = 5
patience = 2

embedding_dim = 256
embedder_layers = [512, embedding_dim]

def get_params():
	return dict(embedder_layers=embedder_layers,
		batch_size=batch_size,
		num_epochs=num_epochs,
		test_interval=test_interval,
		patience=patience)
