import pytest
import warnings
from unittest.mock import PropertyMock

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image

from semanticlens.component_visualization.relevance_based import RelevanceComponentVisualizer, MissingNameWarning, CroppedDataset
from semanticlens.component_visualization.activation_based import ActivationComponentVisualizer
from semanticlens.component_visualization import aggregators

from crp.image import imgify
from crp.attribution import CondAttribution
from crp.helper import load_maximization
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat



@pytest.fixture
def mock_model():
    """Provides a mock model with named modules."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 16, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16*31*31, 2))
    model.layer_names = [str(i) for i in range(len(model))]
    model.name = "mock_model"
    return model


@pytest.fixture
def mock_dataset():
    """Provides a mock dataset of random PIL images."""
    np.random.seed(0)
    # create 2 images 
    images = []
    num_images = 50
    for _ in range(num_images):
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        images.append(img)

    targets = [i % 2 for i in range(num_images)]

    class PILImageDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, targets): 
            self.imgs = imgs
            self.targets = targets
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx): 
            return self.imgs[idx], self.targets[idx]

    return PILImageDataset(images, targets)


def make_transformed_dataset(base_dataset, transform, name=None):
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, base, tfm): 
            self.base, self.tfm = base, tfm
        def __len__(self): 
            return len(self.base)
        def __getitem__(self, idx):
            data, target = self.base[idx] 
            transformed_data = self.tfm(data)
            return transformed_data, target
        
    ds = TransformedDataset(base_dataset, transform)
    
    if name: 
        ds.name = name
    return ds


@pytest.fixture
def mock_dataset_model(mock_dataset):
    """Dataset with model-specific transforms."""
    model_transform = T.Compose([
        T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(35),
        T.ToTensor()
    ])
    return make_transformed_dataset(mock_dataset, model_transform, name="mock_model_dataset")


@pytest.fixture
def mock_dataset_model_full(mock_dataset):
    """Dataset with model-specific transforms including preprocess_fn. (for Activation Component Visualizer)"""
    model_transform = T.Compose([
        T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(35),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return make_transformed_dataset(mock_dataset, model_transform, name="mock_model_dataset")


@pytest.fixture
def mock_dataset_fm(mock_dataset):
    """Dataset with foundation model-specific transforms."""
    fm_transform = T.Compose([
        T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(35),
    ])
    return make_transformed_dataset(mock_dataset, fm_transform)


@pytest.fixture 
def preprocess_fn():
    """Provides preprocessing for FeatureVisualization."""
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Use mocker to create fake versions of the abstract classes
@pytest.fixture
def mock_fm(mocker):
    """Mocks the AbstractVLM (foundation model)."""
    fm = mocker.MagicMock()
    fm.device = "cpu"
    fm.embedding_dim = 128

    def mock_preprocess(pil_images):
        """
        Mock preprocess method that converts PIL images to tensors.
        
        Parameters
        ----------
        pil_images : list[PIL.Image] or PIL.Image
            Single PIL image or list of PIL images
            
        Returns
        -------
        torch.Tensor
            Preprocessed tensor with shape (batch_size, 3, H, W)
        """
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        
        # Convert PIL images to tensors
        transform = T.Compose([
            T.Resize((224, 224)),  # Standardize size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensors = []
        for pil_img in pil_images:
            if isinstance(pil_img, Image.Image):
                tensor = transform(pil_img)
                tensors.append(tensor)
            else:
                # Handle case where input might already be processed
                print(f"Warning: Expected PIL Image, got {type(pil_img)}")
                # Create a dummy tensor
                tensor = torch.randn(3, 224, 224)
                tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(tensors, dim=0)
        return batch_tensor
    
    def mock_encode_image(inputs):
        """
        Mock encode_image method that returns embeddings based on input batch size.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Preprocessed image tensor with shape (batch_size, 3, H, W)
            
        Returns
        -------
        torch.Tensor
            Image embeddings with shape (batch_size, embedding_dim)
        """
        if isinstance(inputs, torch.Tensor):
            batch_size = inputs.shape[0]
        else:
            # Fallback for non-tensor inputs
            batch_size = 1
        
        # Return embeddings with correct batch size
        embeddings = torch.randn(batch_size, fm.embedding_dim)
        return embeddings
    
    # Set up the mock methods
    fm.preprocess.side_effect = mock_preprocess
    fm.encode_image.side_effect = mock_encode_image
    
    # Return self for chained calls like fm.to(device)
    fm.to.return_value = fm
    return fm


def test_initialization(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn):
    """Tests the successful initialization of the visualizer."""
    layer_names = ["2"]
    attribution = CondAttribution(mock_model)
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=["2"],
        num_samples=2,
        preprocess_fn=preprocess_fn,
    )
    assert visualizer.feature_visualization.attribution.model is mock_model
    assert visualizer.layer_names == layer_names
    assert not visualizer.caching


def test_initialization_fails_with_bad_layer(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn):
    """Tests that initialization fails if a layer name is not found."""
    attribution = CondAttribution(mock_model)
    with pytest.raises(ValueError, match="Layer 'bad_layer' not found in model"):
        RelevanceComponentVisualizer(
            attribution=attribution,
            dataset_model=mock_dataset_model,
            dataset_fm=mock_dataset_fm,
            layer_names=["bad_layer"],
            num_samples=2,
            preprocess_fn=preprocess_fn,
        )


def test_missing_name_warning(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn):
    """Tests that a warning is raised if model or dataset lacks a .name attribute when caching."""
    # Remove the .name attribute for the test
    attribution = CondAttribution(mock_model)
    del attribution.model.name

    with pytest.warns(MissingNameWarning, match="Model does not have a name attribute"):
        RelevanceComponentVisualizer(
            attribution=attribution,
            dataset_model=mock_dataset_model,
            dataset_fm=mock_dataset_fm,
            layer_names=["2"],
            num_samples=2,
            preprocess_fn=preprocess_fn,
        )


def test_run_loads_from_cache_if_available(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn, tmp_path, mocker):
    """Tests that the `run` method loads from cache and does not re-compute if a cache file is found."""
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=["2"],
        num_samples=2,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
    )

    # Mock the caching property to return True (cache exists)
    mock_caching = mocker.patch.object(
        type(visualizer),
        'caching',
        new_callable=PropertyMock,
        return_value=True
    )
    
    # Mock collect_results to simulate cached data being loaded
    mock_saved_checkpoints = mocker.patch.object(
        visualizer,
        '_reconstruct_saved_checkpoints',
        return_value={} 
    )

    # Mock the feature_visualization run method to ensure it's NOT called
    mock_run = mocker.patch.object(
        visualizer.feature_visualization,
        'run'
    )
    
    visualizer.run()

    # Verify cache was checked
    mock_caching.assert_called()
    # Verify cached results were collected
    mock_saved_checkpoints.assert_called()
    # Verify that actual computation was NOT performed
    mock_run.assert_not_called()


def test_run_triggers_computation_on_cache_miss(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn, mocker):
    """Tests that the public `run` method triggers the run in feature_visualization when the cache is not found."""
    attribution = CondAttribution(mock_model)
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[],
        num_samples=2,
        preprocess_fn=preprocess_fn,
    )

    # Mock the load method to simulate a cache miss
    mocker.patch.object(
        type(visualizer), 
        'caching', 
        new_callable=PropertyMock,
        return_value=False,
    )

    # Mock the feature_visualization run method to ensure it's called
    mock_run = mocker.patch.object(
        visualizer.feature_visualization,
        'run',
        return_value={}
    )

    # Call the public run method
    visualizer.run(batch_size=2)

    # Assert that run was called
    mock_run.assert_called()


def test_initialization_with_empty_layer_names(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn):
    """Tests that initialization is successful with an empty list of layer names and run completes."""
    attribution = CondAttribution(mock_model)
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[],
        num_samples=2,
        preprocess_fn=preprocess_fn,
    )
    
    assert visualizer.layer_names == []
    # The run should complete without errors and return an empty result
    result = visualizer.run()

    assert all(value == [] for value in result.values()), f"Expected all values to be empty lists, got {result}"

def test_run_with_zero_samples(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn, tmp_path):
    """Tests behavior when num_samples is zero, which should result in empty data."""
    attribution = CondAttribution(mock_model)
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=["2"],
        num_samples=0,
        preprocess_fn=preprocess_fn,
        cache_dir=str(tmp_path),
    )

    # A run should work and the cache should reflect 0 samples collected.
    visualizer.run(batch_size=2)

    sample_ids = visualizer.get_max_reference("2")
    # Tensors should be initialized but have a size of 0 in the collection dimension
    assert sample_ids.shape[1] == 0


def test_cropped_dataset_for_embeddings(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn, tmp_path):
    """Test cropped dataset for embeddings by checking consistency between get_masks_for_max_reference and get_max_reference_examples"""
    num_samples = 2
    layer_name = "2"
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[layer_name],
        num_samples=num_samples,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
    )
    visualizer.run(batch_size=2)

    component_ids = [0]

    # Get cropped references using get_max_reference_examples.
    references_dict = visualizer.get_max_reference_examples(component_ids, layer_name, batch_size=2)
    cropped_references_from_crp = references_dict[component_ids[0]]

    # Get cropped references using masks from get_masks_for_max_reference.
    # Make cropped references for the first component.
    reference_ids = visualizer.get_max_reference(layer_name)[component_ids, :]
    masks = visualizer.get_masks_for_max_reference(component_ids, layer_name, batch_size=2)
    cropped_dataset_fm = CroppedDataset(
        base_dataset=mock_dataset_fm,
        reference_ids=reference_ids[0, :].tolist(),
        masks=masks[component_ids[0]]
    )

    original_width, original_height = mock_dataset_fm[0][0].size  # (H, W)

    # Convert dataset into list of PIL images
    cropped_references_from_masks = []
    for data in cropped_dataset_fm:
        # PIL image -> tensor -> PIL image
        if isinstance(data, (tuple, list)):
            img = data[0]
        else:
            img = data

        # check if masked images are smaller than the original images
        img_width, img_height = img.size
        
        assert img_width < original_width and img_height < original_height, \
            f"Masked image size {img.size} is not smaller than original size {mock_dataset_fm[0][0].size}"
        
        tensor_img = T.ToTensor()(img)
        cropped_references_from_masks.append(
            imgify(tensor_img.detach().cpu()).convert("RGB")
        )

    # Compare each pair of images by converting to numpy arrays and checking equality
    for idx, (img1, img2) in enumerate(zip(cropped_references_from_masks, cropped_references_from_crp)):
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert arr1.shape == arr2.shape, f"Image shapes differ at index {idx}: {arr1.shape} vs {arr2.shape}"
        assert np.allclose(arr1, arr2, atol=1e-9), f"Images differ at index {idx}"


def test_compute_concept_db_one_component(mock_model, mock_fm, mock_dataset_model, mock_dataset_fm, preprocess_fn, tmp_path):
    """Test the computation of the concept database."""
    num_samples = 2
    layer_name = "2"
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[layer_name],
        num_samples=num_samples,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
    )

    # _compute_concept_db runs run internally, so it shouldn't matter whether you do it here or not.
    #visualizer.run(batch_size=2) 

    # one components (int)
    component_ids = {"2": 0}
    concept_db = visualizer._compute_concept_db(mock_fm, component_ids)
    assert all(layer_name == key for key in concept_db.keys()), \
        "Keys don't match expected layer name"
    assert all(isinstance(value, torch.Tensor) for value in concept_db.values()), \
        "All values should be torch.Tensor"
    assert all(value.shape == (1, visualizer.num_samples, mock_fm.embedding_dim) for value in concept_db.values()), \
        "Shapes don't match expected dimensions"


def test_compute_concept_db_two_components(mock_model, mock_fm, mock_dataset_model, mock_dataset_fm, preprocess_fn, tmp_path):
    """Test the computation of the concept database."""
    num_samples = 2
    layer_name = "2"
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[layer_name],
        num_samples=num_samples,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
    )

    visualizer.run(batch_size=2)

    # two components 
    component_ids = {"2": [0, 1]}
    concept_db = visualizer._compute_concept_db(mock_fm, component_ids)
    assert all(layer_name == key for key in concept_db.keys()), \
        "Keys don't match expected layer name"
    assert all(isinstance(value, torch.Tensor) for value in concept_db.values()), \
        "All values should be torch.Tensor"
    assert all(value.shape == (len(component_ids["2"]), visualizer.num_samples, mock_fm.embedding_dim) for value in concept_db.values()), \
        "Shapes don't match expected dimensions"


def test_relevance_visualizer_compared_to_activation_visualizer_max(mock_model, mock_dataset_model, mock_dataset_model_full, mock_dataset_fm, preprocess_fn, tmp_path):
    """Test whether activation and relevance component visualizers produce the same maximally activating references."""
    num_samples = 20
    
    # act comp visualizer
    act_visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset_model_full,
        dataset_fm=mock_dataset_fm,
        layer_names=["2"],
        num_samples=num_samples,
        cache_dir=str(tmp_path),
        aggregate_fn=aggregators.aggregate_conv_max
        
    )
    act_visualizer.run(batch_size=20)
    act_references = act_visualizer.get_max_reference("2")
    act_activations = act_visualizer.actmax_cache.cache["2"].activations

    # rel comp visualizer
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    rel_visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=["2"],
        num_samples=num_samples,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
        aggregate_fn='max'
    )
    rel_visualizer.run(batch_size=20)
    rel_references = rel_visualizer.get_max_reference("2")
    _, rel_activations, _ = load_maximization(rel_visualizer.feature_visualization.ActMax.PATH, "2")
    rel_activations = rel_activations.T

    if not isinstance(act_references, torch.Tensor):
        act_references = torch.from_numpy(act_references.copy())
    if not isinstance(rel_references, torch.Tensor):
        rel_references = torch.from_numpy(rel_references.copy())

    if not isinstance(act_activations, torch.Tensor):
        act_activations = torch.from_numpy(act_activations.copy())
    if not isinstance(rel_activations, torch.Tensor):
        rel_activations = torch.from_numpy(rel_activations.copy())

    # Should have the same shape unless you set num_samples > dataset size
    assert act_references.shape == rel_references.shape, \
        f"Shapes don't match: act={act_references.shape} vs rel={rel_references.shape}"

    assert torch.equal(act_activations, rel_activations), \
        f"Activations don't match:\nact={act_activations}\nrel={rel_activations}, References:\nact={act_references}\nrel={rel_references}"
                         
    assert torch.equal(act_references, rel_references), \
        f"References don't match:\nact={act_references}\nrel={rel_references}"


def test_relevance_visualizer_compared_to_activation_visualizer_sum(mock_model, mock_dataset_model, mock_dataset_model_full, mock_dataset_fm, preprocess_fn, tmp_path):
    """Test whether activation and relevance component visualizers produce the same maximally activating references."""
    num_samples = 10
    layer_name = "2"

    # act comp visualizer
    act_visualizer = ActivationComponentVisualizer(
        model=mock_model,
        dataset_model=mock_dataset_model_full,
        dataset_fm=mock_dataset_fm,
        layer_names=[layer_name],
        num_samples=num_samples,
        cache_dir=str(tmp_path),
        aggregate_fn=aggregators.aggregate_conv_mean
        
    )
    act_visualizer.run(batch_size=20)
    act_references = act_visualizer.get_max_reference(layer_name)
    act_activations = act_visualizer.actmax_cache.cache[layer_name].activations

    # rel comp visualizer
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    rel_visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[layer_name],
        num_samples=num_samples,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
        aggregate_fn='sum'
    )
    rel_visualizer.run(batch_size=20)
    rel_references = rel_visualizer.get_max_reference(layer_name)
    _, rel_activations, _ = load_maximization(rel_visualizer.feature_visualization.ActMax.PATH, "2")
    rel_activations = rel_activations.T

    if not isinstance(act_references, torch.Tensor):
        act_references = torch.from_numpy(act_references.copy())
    if not isinstance(rel_references, torch.Tensor):
        rel_references = torch.from_numpy(rel_references.copy())

    if not isinstance(act_activations, torch.Tensor):
        act_activations = torch.from_numpy(act_activations.copy())
    if not isinstance(rel_activations, torch.Tensor):
        rel_activations = torch.from_numpy(rel_activations.copy())

    # Get the number of nodes in each filter for normalization
    layer = dict(mock_model.named_modules())[layer_name]
    sample_input = mock_dataset_model[0][0].unsqueeze(0)  # add batch dimension
    layers_up_to_target = []
    for name, module in mock_model.named_modules():
        if name == layer_name:
            layers_up_to_target.append(module)
            break
        if name != "":  # Skip the root module
            layers_up_to_target.append(module)
    
    partial_model = nn.Sequential(*layers_up_to_target)
    
    # Forward pass to get the output shape
    with torch.no_grad():
        output = partial_model(sample_input)

    # Get the number of nodes in each filter for normalization
    _, num_filters, height, width = output.shape
    assert num_filters == layer.out_channels, \
        f"Expected {layer.out_channels} filters, but got {num_filters}"
    num_nodes_within_component = height * width

    rel_activations_mean = rel_activations / num_nodes_within_component

    assert act_references.shape == rel_references.shape, \
        f"Shapes don't match: act={act_references.shape} vs rel={rel_references.shape}"

    assert torch.equal(act_activations, rel_activations_mean), \
        f"Activations don't match:\nact={act_activations}\nrel={rel_activations_mean}, \n References:\nact={act_references}\nrel={rel_references}"

    assert torch.equal(act_references, rel_references), \
        f"References don't match:\nact={act_references}\nrel={rel_references}, \n Activations:\nact={act_activations}\nrel={rel_activations_mean}"
    

def test_visualize_components(mock_model, mock_dataset_model, mock_dataset_fm, preprocess_fn, tmp_path):
    """Test the visualize_components method for generating and saving top-k samples."""
    num_samples = 4
    layer_name = "2"
    
    attribution = CondAttribution(mock_model)
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])

    visualizer = RelevanceComponentVisualizer(
        attribution=attribution,
        dataset_model=mock_dataset_model,
        dataset_fm=mock_dataset_fm,
        layer_names=[layer_name],
        num_samples=num_samples,
        preprocess_fn=preprocess_fn,
        composite=composite,
        cache_dir=str(tmp_path),
    )
    visualizer.run(batch_size=20)

    component_ids = torch.tensor([0])
    fname = 'test'
    visualizer.visualize_components(component_ids, layer_name, n_samples=2, fname=fname)

    # check if the image file was created
    component_id_str = "-".join(map(str, component_ids.tolist()))
    fdir = visualizer.storage_dir / "plots"
    fpath = fdir / ((fname + "_" if fname else "") + f"{layer_name}_{component_id_str}.png")
    assert fpath.exists(), f"Expected file at {fpath}, but it does not exist."