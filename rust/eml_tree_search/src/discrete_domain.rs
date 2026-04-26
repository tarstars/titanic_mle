use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use serde::Serialize;

pub type DomainPoint = (f64, f64);
pub type SignatureCounter = HashMap<ValueSignature, u128>;

const DOMAIN_SIZE: usize = 6;

#[derive(Clone, Debug)]
pub struct ValueSignature {
    values: [f64; DOMAIN_SIZE],
}

impl ValueSignature {
    pub fn one() -> Self {
        Self {
            values: [1.0; DOMAIN_SIZE],
        }
    }

    pub fn x0(domain: &[DomainPoint; DOMAIN_SIZE]) -> Self {
        Self {
            values: std::array::from_fn(|index| domain[index].0),
        }
    }

    pub fn x1(domain: &[DomainPoint; DOMAIN_SIZE]) -> Self {
        Self {
            values: std::array::from_fn(|index| domain[index].1),
        }
    }

    pub fn eml(&self, right: &Self) -> Option<Self> {
        let mut out = [0.0; DOMAIN_SIZE];
        for index in 0..DOMAIN_SIZE {
            let left_value = self.values[index];
            let right_value = right.values[index];
            if right_value <= 0.0 {
                return None;
            }

            let exp_left = left_value.exp();
            let ln_right = right_value.ln();
            let value = exp_left - ln_right;
            if !exp_left.is_finite() || !ln_right.is_finite() || !value.is_finite() {
                return None;
            }
            out[index] = value;
        }
        Some(Self { values: out })
    }

    pub fn is_left_safe(&self) -> bool {
        self.values.iter().all(|value| value.exp().is_finite())
    }

    pub fn is_right_safe(&self) -> bool {
        self.values
            .iter()
            .all(|value| *value > 0.0 && value.ln().is_finite())
    }

    pub fn values(&self) -> &[f64; DOMAIN_SIZE] {
        &self.values
    }
}

impl PartialEq for ValueSignature {
    fn eq(&self, other: &Self) -> bool {
        self.values
            .iter()
            .zip(other.values.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

impl Eq for ValueSignature {}

impl Hash for ValueSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.values {
            value.to_bits().hash(state);
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CountSummary {
    pub domain: Vec<DomainPoint>,
    pub exact_height_counts: BTreeMap<usize, u128>,
    pub unique_signature_counts: BTreeMap<usize, usize>,
    pub left_safe_counts: BTreeMap<usize, u128>,
    pub right_safe_counts: BTreeMap<usize, u128>,
    pub total_count_le_5: u128,
}

pub fn sex_embarked_domain() -> [DomainPoint; DOMAIN_SIZE] {
    [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 1.0),
    ]
}

fn leaf_counter(domain: &[DomainPoint; DOMAIN_SIZE]) -> SignatureCounter {
    let mut out = HashMap::new();
    out.insert(ValueSignature::one(), 1);
    out.insert(ValueSignature::x0(domain), 1);
    out.insert(ValueSignature::x1(domain), 1);
    out
}

fn build_exact_signature_counters(max_height: usize) -> Vec<SignatureCounter> {
    let domain = sex_embarked_domain();
    let mut exact = vec![leaf_counter(&domain)];

    for height in 1..=max_height {
        let previous_exact = &exact[height - 1];

        let mut lower: SignatureCounter = HashMap::new();
        for counter in exact.iter().take(height) {
            for (signature, count) in counter {
                *lower.entry(signature.clone()).or_insert(0) += *count;
            }
        }

        let lower_entries: Vec<(ValueSignature, u128)> = lower.into_iter().collect();
        let mut current: SignatureCounter = HashMap::new();

        for (left_signature, left_count) in &lower_entries {
            let left_is_previous = previous_exact.contains_key(left_signature);
            for (right_signature, right_count) in &lower_entries {
                if !(left_is_previous || previous_exact.contains_key(right_signature)) {
                    continue;
                }

                if let Some(output_signature) = left_signature.eml(right_signature) {
                    *current.entry(output_signature).or_insert(0) += left_count * right_count;
                }
            }
        }

        exact.push(current);
    }

    exact
}

pub fn count_valid_sex_embarked_trees_height_le_five() -> CountSummary {
    let exact = build_exact_signature_counters(4);

    let mut exact_height_counts = BTreeMap::new();
    let mut unique_signature_counts = BTreeMap::new();
    let mut left_safe_counts = BTreeMap::new();
    let mut right_safe_counts = BTreeMap::new();

    for (height, counter) in exact.iter().enumerate() {
        let total_count = counter.values().copied().sum();
        let unique_count = counter.len();
        let left_safe_count = counter
            .iter()
            .filter(|(signature, _)| signature.is_left_safe())
            .map(|(_, count)| *count)
            .sum();
        let right_safe_count = counter
            .iter()
            .filter(|(signature, _)| signature.is_right_safe())
            .map(|(_, count)| *count)
            .sum();

        exact_height_counts.insert(height, total_count);
        unique_signature_counts.insert(height, unique_count);
        left_safe_counts.insert(height, left_safe_count);
        right_safe_counts.insert(height, right_safe_count);
    }

    let left_safe_upto_3: u128 = (0..=3)
        .map(|height| left_safe_counts.get(&height).copied().unwrap_or(0))
        .sum();
    let right_safe_upto_4: u128 = (0..=4)
        .map(|height| right_safe_counts.get(&height).copied().unwrap_or(0))
        .sum();
    let left_safe_exact_4 = left_safe_counts.get(&4).copied().unwrap_or(0);
    let right_safe_exact_4 = right_safe_counts.get(&4).copied().unwrap_or(0);

    let exact_height_5_count =
        left_safe_exact_4 * right_safe_upto_4 + left_safe_upto_3 * right_safe_exact_4;
    exact_height_counts.insert(5, exact_height_5_count);

    let total_count_le_5 = exact_height_counts.values().copied().sum();

    CountSummary {
        domain: sex_embarked_domain().to_vec(),
        exact_height_counts,
        unique_signature_counts,
        left_safe_counts,
        right_safe_counts,
        total_count_le_5,
    }
}

#[cfg(test)]
mod tests {
    use super::count_valid_sex_embarked_trees_height_le_five;

    #[test]
    fn known_counts_match_python_reference() {
        let summary = count_valid_sex_embarked_trees_height_le_five();

        assert_eq!(summary.exact_height_counts.get(&0), Some(&3));
        assert_eq!(summary.exact_height_counts.get(&1), Some(&3));
        assert_eq!(summary.exact_height_counts.get(&2), Some(&21));
        assert_eq!(summary.exact_height_counts.get(&3), Some(&543));
        assert_eq!(summary.exact_height_counts.get(&4), Some(&144_123));
        assert_eq!(summary.exact_height_counts.get(&5), Some(&5_355_379_950));

        assert_eq!(summary.unique_signature_counts.get(&3), Some(&538));
        assert_eq!(summary.unique_signature_counts.get(&4), Some(&141_708));

        assert_eq!(summary.left_safe_counts.get(&4), Some(&77_490));
        assert_eq!(summary.right_safe_counts.get(&4), Some(&68_375));
        assert_eq!(summary.total_count_le_5, 5_355_524_643);
    }
}
