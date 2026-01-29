# tests/test_valuations.py
"""Tests for valuation scheme framework."""
import pytest

from src.simulation.valuations import (
    AgentProfile,
    LinearValuationScheme,
    UniformValuationScheme,
    SymmetricValuationScheme,
    FixedValuationScheme,
    get_scheme,
    VALUATION_SCHEMES,
)


class TestAgentProfile:
    def test_profile_creation(self):
        profile = AgentProfile(
            role="buyer",
            valuation=20.0,
            endowment={"money": 100, "good_A": 0}
        )
        assert profile.role == "buyer"
        assert profile.valuation == 20.0
        assert profile.endowment["money"] == 100


class TestLinearValuationScheme:
    def test_default_generates_correct_count(self):
        scheme = LinearValuationScheme()
        profiles = scheme.generate_profiles(n_buyers=2, n_sellers=2)
        assert len(profiles) == 4

    def test_default_buyer_valuations(self):
        """Default: buyers at 20, 22, 24, ..."""
        scheme = LinearValuationScheme()
        profiles = scheme.generate_profiles(n_buyers=3, n_sellers=0)

        valuations = [p.valuation for p in profiles]
        assert valuations == [20.0, 22.0, 24.0]

    def test_default_seller_valuations(self):
        """Default: sellers at 5, 7, 9, ..."""
        scheme = LinearValuationScheme()
        profiles = scheme.generate_profiles(n_buyers=0, n_sellers=3)

        valuations = [p.valuation for p in profiles]
        assert valuations == [5.0, 7.0, 9.0]

    def test_buyers_have_money_endowment(self):
        scheme = LinearValuationScheme()
        profiles = scheme.generate_profiles(n_buyers=2, n_sellers=2)

        buyers = [p for p in profiles if p.role == "buyer"]
        for buyer in buyers:
            assert buyer.endowment["money"] == 100.0
            assert buyer.endowment["good_A"] == 0

    def test_sellers_have_goods_endowment(self):
        scheme = LinearValuationScheme()
        profiles = scheme.generate_profiles(n_buyers=2, n_sellers=2)

        sellers = [p for p in profiles if p.role == "seller"]
        for seller in sellers:
            assert seller.endowment["money"] == 0
            assert seller.endowment["good_A"] == 10.0

    def test_custom_parameters(self):
        scheme = LinearValuationScheme(
            buyer_base=30.0,
            buyer_step=5.0,
            seller_base=10.0,
            seller_step=3.0,
        )
        profiles = scheme.generate_profiles(n_buyers=2, n_sellers=2)

        buyer_vals = [p.valuation for p in profiles if p.role == "buyer"]
        seller_vals = [p.valuation for p in profiles if p.role == "seller"]

        assert buyer_vals == [30.0, 35.0]
        assert seller_vals == [10.0, 13.0]

    def test_name_property(self):
        scheme = LinearValuationScheme()
        assert scheme.name == "linear"

    def test_to_dict_serialization(self):
        scheme = LinearValuationScheme(buyer_base=25.0)
        d = scheme.to_dict()

        assert d["name"] == "linear"
        assert d["params"]["buyer_base"] == 25.0
        assert "buyer_step" in d["params"]
        assert "seller_base" in d["params"]


class TestUniformValuationScheme:
    def test_generates_correct_count(self):
        scheme = UniformValuationScheme(seed=42)
        profiles = scheme.generate_profiles(n_buyers=3, n_sellers=3)
        assert len(profiles) == 6

    def test_valuations_in_range(self):
        scheme = UniformValuationScheme(
            buyer_min=15.0,
            buyer_max=25.0,
            seller_min=5.0,
            seller_max=15.0,
            seed=42
        )
        profiles = scheme.generate_profiles(n_buyers=10, n_sellers=10)

        buyers = [p for p in profiles if p.role == "buyer"]
        sellers = [p for p in profiles if p.role == "seller"]

        for buyer in buyers:
            assert 15.0 <= buyer.valuation <= 25.0

        for seller in sellers:
            assert 5.0 <= seller.valuation <= 15.0

    def test_reproducibility_with_seed(self):
        scheme1 = UniformValuationScheme(seed=42)
        scheme2 = UniformValuationScheme(seed=42)

        profiles1 = scheme1.generate_profiles(n_buyers=3, n_sellers=3)
        profiles2 = scheme2.generate_profiles(n_buyers=3, n_sellers=3)

        vals1 = [p.valuation for p in profiles1]
        vals2 = [p.valuation for p in profiles2]

        assert vals1 == vals2

    def test_different_seeds_different_values(self):
        scheme1 = UniformValuationScheme(seed=42)
        scheme2 = UniformValuationScheme(seed=123)

        profiles1 = scheme1.generate_profiles(n_buyers=3, n_sellers=3)
        profiles2 = scheme2.generate_profiles(n_buyers=3, n_sellers=3)

        vals1 = [p.valuation for p in profiles1]
        vals2 = [p.valuation for p in profiles2]

        assert vals1 != vals2

    def test_name_property(self):
        scheme = UniformValuationScheme()
        assert scheme.name == "uniform"

    def test_to_dict_includes_seed(self):
        scheme = UniformValuationScheme(seed=42)
        d = scheme.to_dict()

        assert d["name"] == "uniform"
        assert d["params"]["seed"] == 42


class TestSymmetricValuationScheme:
    def test_generates_correct_count(self):
        scheme = SymmetricValuationScheme()
        profiles = scheme.generate_profiles(n_buyers=2, n_sellers=2)
        assert len(profiles) == 4

    def test_symmetric_around_equilibrium(self):
        """Buyer highest and seller lowest should be equidistant from eq price."""
        scheme = SymmetricValuationScheme(
            equilibrium_price=10.0,
            spread=10.0,
            step=2.0
        )
        profiles = scheme.generate_profiles(n_buyers=2, n_sellers=2)

        buyers = [p for p in profiles if p.role == "buyer"]
        sellers = [p for p in profiles if p.role == "seller"]

        # Highest buyer: 10 + 5 = 15
        # Second buyer: 15 - 2 = 13
        assert buyers[0].valuation == 15.0
        assert buyers[1].valuation == 13.0

        # Lowest seller: 10 - 5 = 5
        # Second seller: 5 + 2 = 7
        assert sellers[0].valuation == 5.0
        assert sellers[1].valuation == 7.0

    def test_name_property(self):
        scheme = SymmetricValuationScheme()
        assert scheme.name == "symmetric"

    def test_to_dict_serialization(self):
        scheme = SymmetricValuationScheme(equilibrium_price=15.0)
        d = scheme.to_dict()

        assert d["name"] == "symmetric"
        assert d["params"]["equilibrium_price"] == 15.0


class TestFixedValuationScheme:
    def test_uses_exact_valuations(self):
        scheme = FixedValuationScheme(
            buyer_valuations=[25.0, 22.0, 19.0],
            seller_valuations=[8.0, 10.0, 12.0]
        )
        profiles = scheme.generate_profiles(n_buyers=3, n_sellers=3)

        buyer_vals = [p.valuation for p in profiles if p.role == "buyer"]
        seller_vals = [p.valuation for p in profiles if p.role == "seller"]

        assert buyer_vals == [25.0, 22.0, 19.0]
        assert seller_vals == [8.0, 10.0, 12.0]

    def test_raises_on_count_mismatch_buyers(self):
        scheme = FixedValuationScheme(
            buyer_valuations=[25.0, 22.0],
            seller_valuations=[8.0, 10.0]
        )
        with pytest.raises(ValueError, match="Expected 2 buyers"):
            scheme.generate_profiles(n_buyers=3, n_sellers=2)

    def test_raises_on_count_mismatch_sellers(self):
        scheme = FixedValuationScheme(
            buyer_valuations=[25.0, 22.0],
            seller_valuations=[8.0, 10.0]
        )
        with pytest.raises(ValueError, match="Expected 2 sellers"):
            scheme.generate_profiles(n_buyers=2, n_sellers=3)

    def test_name_property(self):
        scheme = FixedValuationScheme(
            buyer_valuations=[20.0],
            seller_valuations=[10.0]
        )
        assert scheme.name == "fixed"

    def test_to_dict_includes_valuations(self):
        scheme = FixedValuationScheme(
            buyer_valuations=[25.0, 22.0],
            seller_valuations=[8.0, 10.0]
        )
        d = scheme.to_dict()

        assert d["name"] == "fixed"
        assert d["params"]["buyer_valuations"] == [25.0, 22.0]
        assert d["params"]["seller_valuations"] == [8.0, 10.0]


class TestGetScheme:
    def test_get_linear_scheme(self):
        scheme = get_scheme("linear")
        assert isinstance(scheme, LinearValuationScheme)

    def test_get_uniform_scheme(self):
        scheme = get_scheme("uniform", seed=42)
        assert isinstance(scheme, UniformValuationScheme)

    def test_get_symmetric_scheme(self):
        scheme = get_scheme("symmetric")
        assert isinstance(scheme, SymmetricValuationScheme)

    def test_get_fixed_scheme(self):
        scheme = get_scheme("fixed", buyer_valuations=[20.0], seller_valuations=[10.0])
        assert isinstance(scheme, FixedValuationScheme)

    def test_raises_on_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unknown scheme"):
            get_scheme("nonexistent")

    def test_passes_kwargs(self):
        scheme = get_scheme("linear", buyer_base=50.0)
        assert scheme.buyer_base == 50.0


class TestSchemeRegistry:
    def test_all_schemes_registered(self):
        assert "linear" in VALUATION_SCHEMES
        assert "uniform" in VALUATION_SCHEMES
        assert "symmetric" in VALUATION_SCHEMES
        assert "fixed" in VALUATION_SCHEMES

    def test_registry_returns_classes(self):
        for name, cls in VALUATION_SCHEMES.items():
            assert hasattr(cls, "generate_profiles")
            assert hasattr(cls, "to_dict")
            assert hasattr(cls, "name")
