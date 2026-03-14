#!/usr/bin/env python3
"""
Example usage script for nitro-libgen output files
Demonstrates how to load, filter, and analyze the generated compound library
"""

import pandas as pd
import os

def demonstrate_library_usage():
    """Show basic operations on the generated compound library"""
    
    # Check if output directory exists
    if not os.path.exists("nitro_library_output"):
        print("❌ Please run generate_nitro_library.py first to create the library")
        return
    
    # 1. LOAD THE MASTER INDEX
    print("\n" + "="*60)
    print("NITRO-LIBGEN: EXAMPLE USAGE")
    print("="*60)
    
    csv_path = "nitro_library_output/compound_library.csv"
    df = pd.read_csv(csv_path)
    print(f"\n📊 Loaded {len(df)} compounds from compound_library.csv")
    print(f"   Columns: {', '.join(df.columns[:8])}...")
    
    # 2. BASIC STATISTICS
    print("\n📈 Basic Statistics:")
    print(f"   Molecular Weight: {df['MW'].astype(float).mean():.1f} ± {df['MW'].astype(float).std():.1f} Da")
    print(f"   LogP: {df['LogP'].astype(float).mean():.2f} ± {df['LogP'].astype(float).std():.2f}")
    print(f"   TPSA: {df['TPSA'].astype(float).mean():.1f} ± {df['TPSA'].astype(float).std():.1f} Å²")
    
    # 3. FILTER FOR DRUG-LIKE COMPOUNDS (Lipinski Rule of 5)
    print("\n💊 Drug-like Filter (Lipinski Rule of 5):")
    drug_like = df[
        (df['MW'].astype(float) < 500) &
        (df['LogP'].astype(float) < 5) &
        (df['HBD'].astype(int) <= 5) &
        (df['HBA'].astype(int) <= 10)
    ]
    print(f"   Compounds passing: {len(drug_like)}/{len(df)} ({len(drug_like)/len(df)*100:.1f}%)")
    
    # 4. FIND MOST SOLUBLE COMPOUNDS (lowest LogP)
    print("\n💧 Top 10 Most Soluble Compounds (lowest LogP):")
    most_soluble = df.nsmallest(10, 'LogP')[['ID', 'Nitro_Replacement', 'Chloro_Replacement', 'LogP']]
    for idx, row in most_soluble.iterrows():
        print(f"   {row['ID']}: LogP={row['LogP']} ({row['Nitro_Replacement']}/{row['Chloro_Replacement']})")
    
    # 5. FIND MOST LIPOPHILIC COMPOUNDS (highest LogP)
    print("\n🕯️ Top 10 Most Lipophilic Compounds (highest LogP):")
    most_lipo = df.nlargest(10, 'LogP')[['ID', 'Nitro_Replacement', 'Chloro_Replacement', 'LogP']]
    for idx, row in most_lipo.iterrows():
        print(f"   {row['ID']}: LogP={row['LogP']} ({row['Nitro_Replacement']}/{row['Chloro_Replacement']})")
    
    # 6. FILTER BY CATEGORY
    print("\n🔬 Compounds by Category:")
    for category in ['acidic', 'basic', 'polar', 'lipophilic']:
        cat_count = len(df[df['Nitro_Category'] == category])
        print(f"   {category.capitalize():12s}: {cat_count} compounds")
    
    # 7. EXPORT FILTERED SUBSET (e.g., for docking)
    print("\n💾 Exporting filtered subset for docking...")
    top_100 = drug_like.head(100)
    top_100.to_csv("top_100_druglike.csv", index=False)
    print(f"   Saved top 100 drug-like compounds to top_100_druglike.csv")
    
    # 8. LOCATE SPECIFIC SDF FILES
    print("\n📁 Accessing Individual SDF Files:")
    example_id = drug_like.iloc[0]['ID']
    sdf_path = f"nitro_library_output/individual_sdfs/{example_id}.sdf"
    print(f"   Example: {example_id}.sdf → {sdf_path}")
    print(f"   (File exists: {os.path.exists(sdf_path)})")
    
    print("\n" + "="*60)
    print("✅ Example complete! Check top_100_druglike.csv for results.")
    print("="*60)

if __name__ == "__main__":
    demonstrate_library_usage()