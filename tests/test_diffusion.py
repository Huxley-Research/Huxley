"""
Tests for the protein structure diffusion generation tools.
"""

import pytest
import asyncio

pytestmark = pytest.mark.asyncio(loop_scope="function")

from huxley.tools.biology.diffusion import (
    # Tools
    generate_protein_structure,
    scaffold_protein_motif,
    design_protein_binder,
    generate_symmetric_assembly,
    validate_protein_structure,
    # Types
    DiffusionBackend,
    DiffusionConfig,
    GeneratedStructure,
    ConditioningType,
    Residue,
    # Backends
    BaseDiffusionBackend,
    FrameDiffBackend,
    MockDiffusionBackend,
    get_backend,
    register_backend,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_pdb():
    """Minimal PDB string for testing."""
    return """HEADER    TEST STRUCTURE
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  SER A   3       7.600   0.000   0.000  1.00  0.00           C
ATOM      4  CA  THR A   4      11.400   0.000   0.000  1.00  0.00           C
ATOM      5  CA  VAL A   5      15.200   0.000   0.000  1.00  0.00           C
END"""


# =============================================================================
# DATA TYPE TESTS
# =============================================================================

class TestDataTypes:
    """Tests for diffusion data types."""
    
    def test_diffusion_backend_enum(self):
        """Test DiffusionBackend enum values."""
        assert DiffusionBackend.MOCK.value == "mock"
        assert DiffusionBackend.RFDIFFUSION.value == "rfdiffusion"
        assert DiffusionBackend.CHROMA.value == "chroma"
        assert DiffusionBackend.FRAMEDIFF.value == "framediff"
    
    def test_conditioning_type_enum(self):
        """Test ConditioningType enum values."""
        assert ConditioningType.NONE.value == "none"
        assert ConditioningType.TEXT.value == "text"
        assert ConditioningType.MOTIF.value == "motif"
        assert ConditioningType.BINDING.value == "binding"
        assert ConditioningType.SYMMETRY.value == "symmetry"
    
    def test_residue_dataclass(self):
        """Test Residue dataclass."""
        res = Residue(
            index=0,
            amino_acid="A",
            x=1.0,
            y=2.0,
            z=3.0,
            fixed=False,
        )
        assert res.index == 0
        assert res.amino_acid == "A"
        assert res.x == 1.0
        assert res.y == 2.0
        assert res.z == 3.0
        assert res.fixed is False
    
    def test_diffusion_config_defaults(self):
        """Test DiffusionConfig default values."""
        config = DiffusionConfig()
        assert config.backend == DiffusionBackend.FRAMEDIFF  # FrameDiff is default
        assert config.num_diffusion_steps == 100  # Optimized for FrameDiff
        assert config.target_length == 100
        assert config.num_samples == 1
        assert config.output_format == "pdb"
        # FrameDiff-specific params
        assert config.t_min == 0.01
        assert config.t_max == 1.0
        assert config.so3_type == "igso3"
    
    def test_diffusion_config_custom(self):
        """Test DiffusionConfig with custom values."""
        config = DiffusionConfig(
            backend=DiffusionBackend.RFDIFFUSION,
            num_diffusion_steps=500,
            target_length=200,
            num_samples=5,
            guidance_scale=5.0,
            seed=42,
        )
        assert config.backend == DiffusionBackend.RFDIFFUSION
        assert config.num_diffusion_steps == 500
        assert config.target_length == 200
        assert config.num_samples == 5
        assert config.guidance_scale == 5.0
        assert config.seed == 42


# =============================================================================
# BACKEND TESTS
# =============================================================================

class TestMockBackend:
    """Tests for the mock diffusion backend."""
    
    @pytest.mark.asyncio
    async def test_backend_is_available(self):
        """Test that mock backend is always available."""
        backend = MockDiffusionBackend()
        assert backend.is_available() is True
    
    @pytest.mark.asyncio
    async def test_backend_load_model(self):
        """Test model loading."""
        backend = MockDiffusionBackend()
        await backend.load_model()
        assert backend._loaded is True
    
    @pytest.mark.asyncio
    async def test_backend_generate_single(self):
        """Test generating a single structure."""
        backend = MockDiffusionBackend()
        await backend.load_model()
        
        config = DiffusionConfig(
            target_length=50,
            num_samples=1,
            seed=42,
        )
        
        structures = await backend.generate(config)
        
        assert len(structures) == 1
        struct = structures[0]
        assert struct.length == 50
        assert len(struct.sequence) == 50
        assert len(struct.coordinates) == 50
        assert struct.pdb_string is not None
        assert struct.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_backend_generate_multiple(self):
        """Test generating multiple structures."""
        backend = MockDiffusionBackend()
        await backend.load_model()
        
        config = DiffusionConfig(
            target_length=60,
            num_samples=5,
        )
        
        structures = await backend.generate(config)
        
        assert len(structures) == 5
        for struct in structures:
            assert struct.length == 60
    
    @pytest.mark.asyncio
    async def test_backend_with_conditioning(self):
        """Test generation with conditioning."""
        backend = MockDiffusionBackend()
        await backend.load_model()
        
        config = DiffusionConfig(target_length=50, backend=DiffusionBackend.MOCK)
        conditioning = {
            "type": "text",
            "text": "alpha helical bundle",
        }
        
        structures = await backend.generate(config, conditioning)
        
        assert len(structures) == 1
        assert structures[0].generation_params["conditioning"] == conditioning
    
    @pytest.mark.asyncio
    async def test_backend_seed_reproducibility(self):
        """Test that same seed produces same results."""
        backend = MockDiffusionBackend()
        await backend.load_model()
        
        config = DiffusionConfig(
            backend=DiffusionBackend.MOCK,
            target_length=50,
            num_samples=1,
            seed=12345,
        )
        
        structures1 = await backend.generate(config)
        structures2 = await backend.generate(config)
        
        assert structures1[0].sequence == structures2[0].sequence


class TestFrameDiffBackend:
    """Tests for the FrameDiff backend (recommended)."""
    
    @pytest.mark.asyncio
    async def test_framediff_is_available(self):
        """Test that FrameDiff backend is available."""
        backend = FrameDiffBackend()
        assert backend.is_available() is True
    
    @pytest.mark.asyncio
    async def test_framediff_load_model(self):
        """Test FrameDiff model loading."""
        backend = FrameDiffBackend()
        await backend.load_model()
        assert backend._loaded is True
    
    @pytest.mark.asyncio
    async def test_framediff_generate_single(self):
        """Test generating a single structure with FrameDiff."""
        backend = FrameDiffBackend()
        await backend.load_model()
        
        config = DiffusionConfig(
            backend=DiffusionBackend.FRAMEDIFF,
            target_length=60,
            num_samples=1,
            seed=42,
        )
        
        structures = await backend.generate(config)
        
        assert len(structures) == 1
        struct = structures[0]
        assert struct.length == 60
        assert len(struct.sequence) == 60
        assert struct.backend_used == "framediff"
        assert struct.pdb_string is not None
        # FrameDiff generates full backbone (N, CA, C, O)
        assert "FRAMEDIFF" in struct.pdb_string
        assert "  N  " in struct.pdb_string
        assert "  CA " in struct.pdb_string
        assert "  C  " in struct.pdb_string
        assert "  O  " in struct.pdb_string
    
    @pytest.mark.asyncio
    async def test_framediff_with_helix_conditioning(self):
        """Test FrameDiff generation with helix conditioning."""
        backend = FrameDiffBackend()
        await backend.load_model()
        
        config = DiffusionConfig(
            backend=DiffusionBackend.FRAMEDIFF,
            target_length=50,
            num_samples=1,
        )
        conditioning = {
            "type": "text",
            "text": "alpha helical bundle",
        }
        
        structures = await backend.generate(config, conditioning)
        
        assert len(structures) == 1
        assert structures[0].generation_params["conditioning"] == conditioning
    
    @pytest.mark.asyncio
    async def test_framediff_confidence_scores(self):
        """Test that FrameDiff generates confidence scores."""
        backend = FrameDiffBackend()
        await backend.load_model()
        
        config = DiffusionConfig(
            backend=DiffusionBackend.FRAMEDIFF,
            target_length=50,
        )
        
        structures = await backend.generate(config)
        struct = structures[0]
        
        assert struct.confidence_score > 0
        assert len(struct.per_residue_confidence) == 50
        for conf in struct.per_residue_confidence:
            assert 0 <= conf <= 1


class TestBackendRegistry:
    """Tests for backend registry functions."""
    
    def test_get_framediff_backend(self):
        """Test getting the FrameDiff backend (default)."""
        backend = get_backend(DiffusionBackend.FRAMEDIFF)
        assert isinstance(backend, FrameDiffBackend)
    
    def test_get_mock_backend(self):
        """Test getting the mock backend."""
        backend = get_backend(DiffusionBackend.MOCK)
        assert isinstance(backend, MockDiffusionBackend)
    
    def test_get_backend_caches_instance(self):
        """Test that get_backend returns the same instance."""
        backend1 = get_backend(DiffusionBackend.FRAMEDIFF)
        backend2 = get_backend(DiffusionBackend.FRAMEDIFF)
        assert backend1 is backend2
    
    def test_register_custom_backend(self):
        """Test registering a custom backend."""
        class CustomBackend(BaseDiffusionBackend):
            async def generate(self, config, conditioning=None):
                return []
            
            async def load_model(self, model_path=None):
                pass
            
            def is_available(self):
                return True
        
        # Register won't error, just adds to registry
        register_backend(DiffusionBackend.FRAMEDIFF, CustomBackend)


# =============================================================================
# TOOL FUNCTION TESTS
# =============================================================================

class TestGenerateProteinStructure:
    """Tests for the generate_protein_structure tool."""
    
    @pytest.mark.asyncio
    async def test_basic_generation_framediff(self):
        """Test basic structure generation with FrameDiff (default)."""
        result = await generate_protein_structure(
            target_length=50,
            num_samples=1,
        )
        
        assert result["success"] is True
        assert result["num_generated"] == 1
        assert len(result["structures"]) == 1
        assert result["config"]["backend"] == "framediff"
        
        struct = result["structures"][0]
        assert struct["length"] == 50
        assert len(struct["sequence"]) == 50
        assert struct["pdb"] is not None
        assert "FRAMEDIFF" in struct["pdb"]
    
    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Test basic structure generation with mock backend."""
        result = await generate_protein_structure(
            target_length=50,
            num_samples=1,
            backend="mock",
        )
        
        assert result["success"] is True
        assert result["num_generated"] == 1
        assert len(result["structures"]) == 1
        
        struct = result["structures"][0]
        assert struct["length"] == 50
        assert len(struct["sequence"]) == 50
        assert struct["pdb"] is not None
    
    @pytest.mark.asyncio
    async def test_generation_with_text_conditioning(self):
        """Test generation with text conditioning."""
        result = await generate_protein_structure(
            target_length=75,
            conditioning_text="beta barrel membrane protein",
            backend="mock",
        )
        
        assert result["success"] is True
        assert result["config"]["conditioning"] == "beta barrel membrane protein"
    
    @pytest.mark.asyncio
    async def test_generation_multiple_samples(self):
        """Test generating multiple samples."""
        result = await generate_protein_structure(
            target_length=40,
            num_samples=3,
            backend="mock",
        )
        
        assert result["success"] is True
        assert result["num_generated"] == 3
        assert len(result["structures"]) == 3
    
    @pytest.mark.asyncio
    async def test_generation_with_validation(self):
        """Test that validation is performed by default."""
        result = await generate_protein_structure(
            target_length=50,
            validate=True,
            backend="mock",
        )
        
        assert result["success"] is True
        assert result["validation"] is not None
        assert len(result["validation"]) == 1
    
    @pytest.mark.asyncio
    async def test_generation_without_validation(self):
        """Test disabling validation."""
        result = await generate_protein_structure(
            target_length=50,
            validate=False,
            backend="mock",
        )
        
        assert result["success"] is True
        assert result["validation"] is None
    
    @pytest.mark.asyncio
    async def test_input_bounds_enforcement(self):
        """Test that input bounds are enforced."""
        # Length too small - should be clamped to 50
        result = await generate_protein_structure(
            target_length=10,
            backend="mock",
        )
        assert result["success"] is True
        assert result["structures"][0]["length"] == 50
        
        # Length too large - should be clamped to 500
        result = await generate_protein_structure(
            target_length=1000,
            backend="mock",
        )
        assert result["success"] is True
        assert result["structures"][0]["length"] == 500


class TestScaffoldProteinMotif:
    """Tests for the scaffold_protein_motif tool."""
    
    @pytest.mark.asyncio
    async def test_basic_scaffolding(self, sample_pdb):
        """Test basic motif scaffolding."""
        result = await scaffold_protein_motif(
            motif_pdb=sample_pdb,
            motif_residues=[10, 20, 30],
            scaffold_length=100,
        )
        
        assert result["success"] is True
        assert result["num_generated"] == 1
        assert result["motif_info"]["fixed_residues"] == [10, 20, 30]
        assert result["motif_info"]["scaffold_length"] == 100
    
    @pytest.mark.asyncio
    async def test_scaffolding_multiple_designs(self, sample_pdb):
        """Test generating multiple scaffold designs."""
        result = await scaffold_protein_motif(
            motif_pdb=sample_pdb,
            motif_residues=[5, 15],
            scaffold_length=80,
            num_samples=3,
        )
        
        assert result["success"] is True
        assert result["num_generated"] == 3


class TestDesignProteinBinder:
    """Tests for the design_protein_binder tool."""
    
    @pytest.mark.asyncio
    async def test_basic_binder_design(self, sample_pdb):
        """Test basic binder design."""
        result = await design_protein_binder(
            target_pdb=sample_pdb,
            target_chain="A",
            binder_length=80,
        )
        
        assert result["success"] is True
        assert result["num_designs"] == 3  # Default
        assert len(result["binder_designs"]) == 3
        
        for design in result["binder_designs"]:
            assert "binding_prediction" in design
            assert "predicted_affinity_nm" in design["binding_prediction"]
    
    @pytest.mark.asyncio
    async def test_binder_with_hotspots(self, sample_pdb):
        """Test binder design with hotspot residues."""
        result = await design_protein_binder(
            target_pdb=sample_pdb,
            target_chain="A",
            hotspot_residues=[1, 2, 3],
            num_designs=2,
        )
        
        assert result["success"] is True
        assert result["target_info"]["hotspot_residues"] == [1, 2, 3]


class TestGenerateSymmetricAssembly:
    """Tests for the generate_symmetric_assembly tool."""
    
    @pytest.mark.asyncio
    async def test_cyclic_symmetry(self):
        """Test cyclic symmetric assembly generation."""
        result = await generate_symmetric_assembly(
            symmetry_type="cyclic",
            symmetry_order=3,
            subunit_length=50,
        )
        
        assert result["success"] is True
        assert result["symmetry_info"]["type"] == "cyclic"
        assert result["symmetry_info"]["order"] == 3
        assert result["symmetry_info"]["num_subunits"] == 3
    
    @pytest.mark.asyncio
    async def test_dihedral_symmetry(self):
        """Test dihedral symmetric assembly generation."""
        result = await generate_symmetric_assembly(
            symmetry_type="dihedral",
            symmetry_order=4,
            subunit_length=60,
        )
        
        assert result["success"] is True
        assert result["symmetry_info"]["type"] == "dihedral"
        assert result["symmetry_info"]["num_subunits"] == 8  # D4 = 8 subunits
    
    @pytest.mark.asyncio
    async def test_tetrahedral_symmetry(self):
        """Test tetrahedral symmetric assembly generation."""
        result = await generate_symmetric_assembly(
            symmetry_type="tetrahedral",
            subunit_length=40,
        )
        
        assert result["success"] is True
        assert result["symmetry_info"]["num_subunits"] == 12
    
    @pytest.mark.asyncio
    async def test_invalid_symmetry_type(self):
        """Test error handling for invalid symmetry type."""
        result = await generate_symmetric_assembly(
            symmetry_type="invalid_symmetry",
        )
        
        assert result["success"] is False
        assert "error" in result


class TestValidateProteinStructure:
    """Tests for the validate_protein_structure tool."""
    
    @pytest.mark.asyncio
    async def test_valid_structure(self, sample_pdb):
        """Test validating a valid structure."""
        result = await validate_protein_structure(pdb_string=sample_pdb)
        
        assert result["success"] is True
        assert result["sequence"] == "AGSTV"
        assert result["length"] == 5
        assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_empty_pdb(self):
        """Test validating an empty PDB."""
        result = await validate_protein_structure(pdb_string="")
        
        assert result["success"] is False
        assert result["valid"] is False
    
    @pytest.mark.asyncio
    async def test_invalid_pdb_format(self):
        """Test validating malformed PDB."""
        result = await validate_protein_structure(
            pdb_string="This is not a PDB file"
        )
        
        assert result["success"] is False
        assert result["valid"] is False
    
    @pytest.mark.asyncio
    async def test_structure_with_clashes(self):
        """Test validation detects clashes."""
        # PDB with atoms too close together
        pdb_with_clashes = """HEADER    TEST
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  SER A   3       0.500   0.000   0.000  1.00  0.00           C
END"""
        
        result = await validate_protein_structure(pdb_string=pdb_with_clashes)
        
        assert result["success"] is True
        assert "metrics" in result
        assert "clash_count" in result["metrics"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the diffusion module."""
    
    @pytest.mark.asyncio
    async def test_generate_and_validate(self):
        """Test generating a structure and then validating it."""
        # Generate
        gen_result = await generate_protein_structure(
            target_length=50,
            backend="mock",
            validate=False,
        )
        
        assert gen_result["success"] is True
        pdb = gen_result["structures"][0]["pdb"]
        
        # Validate
        val_result = await validate_protein_structure(pdb_string=pdb)
        
        assert val_result["success"] is True
        assert val_result["length"] == 50
    
    @pytest.mark.asyncio
    async def test_workflow_binder_design(self, sample_pdb):
        """Test a complete binder design workflow."""
        # 1. Design binders
        binders = await design_protein_binder(
            target_pdb=sample_pdb,
            binder_length=60,
            num_designs=2,
        )
        
        assert binders["success"] is True
        
        # 2. Validate each binder
        for design in binders["binder_designs"]:
            val_result = await validate_protein_structure(
                pdb_string=design["pdb"]
            )
            assert val_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_all_tools_run(self, sample_pdb):
        """Smoke test that all tools can be invoked."""
        # This ensures no import or runtime errors
        results = await asyncio.gather(
            generate_protein_structure(target_length=30, backend="mock"),
            scaffold_protein_motif(motif_pdb=sample_pdb, motif_residues=[1, 2]),
            design_protein_binder(target_pdb=sample_pdb, num_designs=1),
            generate_symmetric_assembly(symmetry_type="cyclic", symmetry_order=2),
            validate_protein_structure(pdb_string=sample_pdb),
        )
        
        for result in results:
            assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
