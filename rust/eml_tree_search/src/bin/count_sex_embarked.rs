use eml_tree_search::count_valid_sex_embarked_trees_height_le_five;

fn main() {
    let summary = count_valid_sex_embarked_trees_height_le_five();
    let json = serde_json::to_string_pretty(&summary).expect("summary should serialize");
    println!("{json}");
}
