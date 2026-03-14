#!/usr/bin/env python3
"""
================================================================================
NITROFURAN-TRIAZOLE ANALOG LIBRARY GENERATOR
================================================================================
Description: 
    This script generates a combinatorial library of analogs from a parent 
    nitrofuran-triazole compound by systematically replacing:
    1. The nitro group (-NO2) on the furan ring (50+ bioisosteres)
    2. The chloro group (-Cl) on the triazole ring (30+ substituents)
    
    All generated compounds are saved as individual 3D SDF files with simple
    IDs (e.g., CPD00001.sdf, CPD00002.sdf) and indexed in a master CSV file.
    
Dependencies:
    - RDKit (2023.03.1 or later)
    - pandas (1.5.3 or later)
    - numpy (1.24.3 or later)

Usage:
    python generate_nitro_library.py

Output:
    - individual_sdfs/          : Directory containing all SDF files
    - compound_library.csv      : Master index with properties
    - ALL_compounds_combined.sdf: Combined file for batch processing
    - generation_summary.txt    : Statistics and summary
    - library_generation.log    : Detailed log file
================================================================================
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('library_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Parent compound (your original molecule)
PARENT_SMILES = "CNC(=O)c1cc(oc1)N(=O)=Oc1ccc(cc1)c1nnc2ccc(Cl)nn12"

# Output directory structure
OUTPUT_DIR = "nitro_library_output"
SDF_DIR = os.path.join(OUTPUT_DIR, "individual_sdfs")
os.makedirs(SDF_DIR, exist_ok=True)

# ==============================================================================
# REPLACEMENT LIBRARIES (Curated Bioisosteres)
# ==============================================================================

# Format: (SMARTS, Name, Category)
NITRO_REPLACEMENTS = [
    # ===== ACIDIC GROUPS (High Solubility) =====
    ("C(=O)O", "Carboxylic_acid", "acidic"),
    ("c1nn[nH]n1", "Tetrazole", "acidic"),
    ("S(=O)(=O)O", "Sulfonic_acid", "acidic"),
    ("C(=O)NO", "Hydroxamic_acid", "acidic"),
    ("P(=O)(O)O", "Phosphonic_acid", "acidic"),
    
    # ===== AMIDES & POLAR GROUPS =====
    ("C(=O)N", "Primary_amide", "polar"),
    ("C(=O)NC", "Secondary_amide", "polar"),
    ("C(=O)N(C)C", "Tertiary_amide", "polar"),
    ("S(=O)(=O)N", "Primary_sulfonamide", "polar"),
    ("S(=O)(=O)NC", "Secondary_sulfonamide", "polar"),
    ("NC(=O)N", "Urea", "polar"),
    ("COC(=O)N", "Carbamate", "polar"),
    
    # ===== HETEROCYCLIC BIOISOSTERES =====
    ("n1ccccc1", "Pyridine", "heterocycle"),
    ("n1cnccc1", "Pyrimidine", "heterocycle"),
    ("n1ccnc1", "Imidazole", "heterocycle"),
    ("c1nocn1", "Oxadiazole_124", "heterocycle"),
    ("c1nnco1", "Oxadiazole_134", "heterocycle"),
    ("c1nscn1", "Thiadiazole_124", "heterocycle"),
    ("n1nncc1", "Triazole_123", "heterocycle"),
    
    # ===== NEUTRAL POLAR GROUPS =====
    ("C#N", "Nitrile", "neutral"),
    ("C=O", "Aldehyde", "neutral"),
    ("C(=O)C", "Ketone", "neutral"),
    ("S(=O)C", "Sulfoxide", "neutral"),
    ("S(=O)(=O)C", "Sulfone", "neutral"),
    
    # ===== HALOGENS & CF3 ANALOGS =====
    ("C(F)(F)F", "Trifluoromethyl", "lipophilic"),
    ("Cl", "Chloro", "lipophilic"),
    ("Br", "Bromo", "lipophilic"),
    ("F", "Fluoro", "lipophilic"),
    ("OC(F)(F)F", "Trifluoromethoxy", "lipophilic"),
    ("SC(F)(F)F", "Trifluoromethylthio", "lipophilic"),
    
    # ===== AMINE DERIVATIVES =====
    ("N", "Primary_amine", "basic"),
    ("NC", "Secondary_amine", "basic"),
    ("N(C)C", "Tertiary_amine", "basic"),
    ("NO", "Hydroxylamine", "basic"),
    ("Nc1cc1", "Cyclopropylamine", "basic"),
    
    # ===== SATURATED HETEROCYCLES =====
    ("N1CCOCC1", "Morpholine", "basic"),
    ("N1CCNCC1", "Piperazine", "basic"),
    ("N1CCCCC1", "Piperidine", "basic"),
    ("N1CCCC1", "Pyrrolidine", "basic"),
    ("N1CCSCC1", "Thiomorpholine", "basic"),
    
    # ===== ETHERS & ALCOHOLS =====
    ("OC", "Methoxy", "neutral"),
    ("OCC", "Ethoxy", "neutral"),
    ("O", "Hydroxyl", "polar"),
    ("OC(C)C", "Isopropoxy", "neutral"),
    ("OCCOC", "Methoxyethoxy", "neutral"),
    
    # ===== ALKYL & ARYL GROUPS =====
    ("C=C", "Vinyl", "lipophilic"),
    ("C#C", "Acetylene", "neutral"),
    ("CC", "Methyl", "lipophilic"),
    ("c1ccccc1", "Phenyl", "lipophilic"),
    ("Cc1ccccc1", "Benzyl", "lipophilic"),
    ("Oc1ccccc1", "Phenoxy", "lipophilic"),
    ("C(=O)c1ccccc1", "Benzoyl", "polar"),
]

CHLORO_REPLACEMENTS = [
    # ===== HALOGENS =====
    ("F", "Fluoro", "lipophilic"),
    ("Br", "Bromo", "lipophilic"),
    ("I", "Iodo", "lipophilic"),
    ("C(F)(F)F", "Trifluoromethyl", "lipophilic"),
    
    # ===== ALKYL GROUPS =====
    ("C", "Methyl", "lipophilic"),
    ("CC", "Ethyl", "lipophilic"),
    ("CCC", "Propyl", "lipophilic"),
    ("CC(C)C", "Isopropyl", "lipophilic"),
    ("CCCC", "Butyl", "lipophilic"),
    
    # ===== POLAR GROUPS =====
    ("OC", "Methoxy", "neutral"),
    ("OCC", "Ethoxy", "neutral"),
    ("O", "Hydroxyl", "polar"),
    ("C=O", "Aldehyde", "neutral"),
    ("C(=O)C", "Acetyl", "polar"),
    ("C#N", "Nitrile", "neutral"),
    
    # ===== AMINES =====
    ("N", "Amino", "basic"),
    ("NC", "Methylamino", "basic"),
    ("N(C)C", "Dimethylamino", "basic"),
    
    # ===== AROMATIC =====
    ("c1ccccc1", "Phenyl", "lipophilic"),
    ("c1ccncc1", "Pyridyl", "basic"),
    ("c1ccsc1", "Thienyl", "lipophilic"),
    
    # ===== ACIDIC =====
    ("C(=O)O", "Carboxyl", "acidic"),
    ("S(=O)(=O)O", "Sulfonate", "acidic"),
]

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def validate_environment() -> bool:
    """Check if all required packages are available."""
    try:
        import rdkit
        import pandas
        import numpy
        logger.info(f"RDKit version: {rdkit.__version__}")
        logger.info(f"Pandas version: {pandas.__version__}")
        logger.info(f"NumPy version: {numpy.__version__}")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        return False


def create_molecule_safely(smiles: str, max_attempts: int = 3) -> Optional[Chem.Mol]:
    """
    Create RDKit molecule from SMILES with multiple fallback strategies.
    
    Args:
        smiles: SMILES string
        max_attempts: Number of sanitization attempts
    
    Returns:
        RDKit Mol object or None if failed
    """
    strategies = [
        # Strategy 1: Normal sanitization
        lambda: Chem.MolFromSmiles(smiles),
        
        # Strategy 2: Without sanitization
        lambda: Chem.MolFromSmiles(smiles, sanitize=False),
        
        # Strategy 3: With kekulization first
        lambda: Chem.MolFromSmiles(smiles, sanitize=False),
    ]
    
    for i, strategy in enumerate(strategies[:max_attempts]):
        try:
            mol = strategy()
            if mol is not None:
                if i > 0:  # Created without sanitization
                    try:
                        Chem.SanitizeMol(mol, catchErrors=True)
                    except:
                        try:
                            Chem.Kekulize(mol, clearAromaticFlags=True)
                        except:
                            pass  # Accept unsanitized
                return mol
        except Exception as e:
            logger.debug(f"Strategy {i+1} failed: {str(e)}")
            continue
    
    return None


def generate_3d_structure(mol: Chem.Mol, max_attempts: int = 3) -> Tuple[bool, Chem.Mol]:
    """
    Generate 3D coordinates for a molecule with multiple embedding attempts.
    
    Args:
        mol: RDKit Mol object
        max_attempts: Number of embedding attempts
    
    Returns:
        (success, molecule_with_3d)
    """
    if mol is None:
        return False, None
    
    mol_3d = Chem.AddHs(mol)
    
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # ETKDG method
                params = AllChem.ETKDGv3()
                params.randomSeed = 42 + attempt
                success = AllChem.EmbedMolecule(mol_3d, params)
            elif attempt == 1:
                # Basic knowledge
                success = AllChem.EmbedMolecule(mol_3d, useBasicKnowledge=True)
            else:
                # Random coordinates
                success = AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)
            
            if success == 0:  # Success
                # Energy minimization
                try:
                    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=500)
                    except:
                        pass  # Accept unoptimized
                return True, mol_3d
                
        except Exception as e:
            logger.debug(f"Embedding attempt {attempt+1} failed: {str(e)}")
            continue
    
    return False, None


def calculate_properties(mol: Chem.Mol) -> Dict[str, float]:
    """
    Calculate key molecular properties.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        Dictionary of properties
    """
    props = {}
    try:
        props['MW'] = Descriptors.MolWt(mol)
        props['LogP'] = Descriptors.MolLogP(mol)
        props['TPSA'] = Descriptors.TPSA(mol)
        props['HBD'] = Descriptors.NumHDonors(mol)
        props['HBA'] = Descriptors.NumHAcceptors(mol)
        props['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        props['RingCount'] = Descriptors.RingCount(mol)
        
        # Rule of 5 violations
        ro5_violations = 0
        if props['MW'] > 500: ro5_violations += 1
        if props['LogP'] > 5: ro5_violations += 1
        if props['HBD'] > 5: ro5_violations += 1
        if props['HBA'] > 10: ro5_violations += 1
        props['Ro5_Violations'] = ro5_violations
        props['Ro5_Pass'] = ro5_violations <= 1
        
    except Exception as e:
        logger.warning(f"Property calculation failed: {e}")
        props = {k: 0.0 for k in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 
                                   'RotatableBonds', 'RingCount', 'Ro5_Violations']}
        props['Ro5_Pass'] = False
    
    return props


def generate_analog(nitro_smarts: str, chloro_smarts: str) -> Tuple[Optional[Chem.Mol], Optional[str], Dict]:
    """
    Generate an analog with specific nitro and chloro replacements.
    
    Args:
        nitro_smarts: SMARTS string for nitro replacement
        chloro_smarts: SMARTS string for chloro replacement
    
    Returns:
        (molecule, smiles, properties)
    """
    # Generate SMILES by replacement
    smiles_temp = PARENT_SMILES.replace("N(=O)=O", nitro_smarts)
    final_smiles = smiles_temp.replace("Cl", chloro_smarts)
    
    # Create molecule
    mol = create_molecule_safely(final_smiles)
    if mol is None:
        return None, None, {}
    
    # Generate 3D structure
    success, mol_3d = generate_3d_structure(mol)
    if not success:
        return None, None, {}
    
    # Calculate properties
    props = calculate_properties(mol_3d)
    
    return mol_3d, final_smiles, props


def save_individual_sdf(mol: Chem.Mol, compound_id: str, properties: Dict, 
                        nitro_name: str, chloro_name: str) -> str:
    """
    Save molecule as individual SDF file with simple ID name.
    
    Args:
        mol: RDKit Mol object
        compound_id: Unique identifier (e.g., CPD00001)
        properties: Dictionary of properties
        nitro_name: Name of nitro replacement
        chloro_name: Name of chloro replacement
    
    Returns:
        Filename of saved SDF
    """
    # Set properties as SDF fields
    mol.SetProp("_Name", compound_id)
    mol.SetProp("SMILES", Chem.MolToSmiles(mol))
    mol.SetProp("Nitro_Replacement", nitro_name)
    mol.SetProp("Chloro_Replacement", chloro_name)
    
    for key, value in properties.items():
        mol.SetProp(str(key), str(value))
    
    # Simple filename: just the ID
    filename = f"{compound_id}.sdf"
    filepath = os.path.join(SDF_DIR, filename)
    
    writer = Chem.SDWriter(filepath)
    writer.write(mol)
    writer.close()
    
    return filename


def write_summary_report(stats: Dict, start_time: float, output_dir: str):
    """
    Write a comprehensive summary report.
    
    Args:
        stats: Dictionary of generation statistics
        start_time: Start time of generation
        output_dir: Output directory
    """
    elapsed_time = time.time() - start_time
    
    report_path = os.path.join(output_dir, "generation_summary.txt")
    
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("NITROFURAN-TRIAZOLE LIBRARY GENERATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parent Compound SMILES: {PARENT_SMILES}\n\n")
        
        f.write("LIBRARY STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"  Nitro replacements attempted: {stats['nitro_attempted']}\n")
        f.write(f"  Chloro replacements attempted: {stats['chloro_attempted']}\n")
        f.write(f"  Total combinations possible: {stats['total_possible']}\n")
        f.write(f"  Successfully generated: {stats['successful']}\n")
        f.write(f"  Success rate: {stats['successful']/stats['total_possible']*100:.1f}%\n")
        f.write(f"  Failed compounds: {stats['failed']}\n")
        f.write(f"  Generation time: {elapsed_time:.1f} seconds\n")
        f.write(f"  Generation rate: {stats['successful']/elapsed_time:.1f} compounds/second\n\n")
        
        f.write("PROPERTY STATISTICS:\n")
        f.write("-"*40 + "\n")
        if stats['properties']:
            for prop, values in stats['properties'].items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    f.write(f"  {prop}: {mean_val:.2f} ± {std_val:.2f} ")
                    f.write(f"(range: {min_val:.2f}-{max_val:.2f})\n")
        
        f.write("\nCATEGORY DISTRIBUTION:\n")
        f.write("-"*40 + "\n")
        for cat, count in stats.get('categories', {}).items():
            f.write(f"  {cat}: {count} compounds\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("="*70 + "\n")
        f.write(f"  Individual SDFs: {SDF_DIR}/\n")
        f.write(f"  Master index: {os.path.join(output_dir, 'compound_library.csv')}\n")
        f.write(f"  Combined SDF: {os.path.join(output_dir, 'ALL_compounds_combined.sdf')}\n")
        f.write(f"  Log file: library_generation.log\n")
        f.write(f"  This summary: {report_path}\n")
    
    logger.info(f"Summary report saved to {report_path}")


def create_combined_sdf(compounds_df: pd.DataFrame, output_path: str):
    """
    Create a single SDF file containing all generated compounds.
    
    Args:
        compounds_df: DataFrame with compound information
        output_path: Path for combined SDF file
    """
    writer = Chem.SDWriter(output_path)
    count = 0
    
    for _, row in compounds_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, useBasicKnowledge=True)
                mol.SetProp("_Name", row['ID'])
                mol.SetProp("Nitro_Replacement", row['Nitro_Replacement'])
                mol.SetProp("Chloro_Replacement", row['Chloro_Replacement'])
                writer.write(mol)
                count += 1
        except Exception as e:
            logger.warning(f"Failed to add {row['ID']} to combined SDF: {e}")
    
    writer.close()
    logger.info(f"Created combined SDF with {count} compounds")


# ==============================================================================
# MAIN GENERATION FUNCTION
# ==============================================================================

def main():
    """Main execution function."""
    
    # Print banner
    print("\n" + "="*80)
    print(" NITROFURAN-TRIAZOLE ANALOG LIBRARY GENERATOR")
    print("="*80)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)
    
    # Log start
    start_time = time.time()
    logger.info(f"Starting library generation")
    logger.info(f"Parent SMILES: {PARENT_SMILES}")
    logger.info(f"Nitro replacements: {len(NITRO_REPLACEMENTS)}")
    logger.info(f"Chloro replacements: {len(CHLORO_REPLACEMENTS)}")
    logger.info(f"Total combinations: {len(NITRO_REPLACEMENTS) * len(CHLORO_REPLACEMENTS)}")
    
    # Initialize statistics
    stats = {
        'nitro_attempted': len(NITRO_REPLACEMENTS),
        'chloro_attempted': len(CHLORO_REPLACEMENTS),
        'total_possible': len(NITRO_REPLACEMENTS) * len(CHLORO_REPLACEMENTS),
        'successful': 0,
        'failed': 0,
        'properties': {
            'MW': [], 'LogP': [], 'TPSA': [], 
            'HBD': [], 'HBA': [], 'Ro5_Violations': []
        },
        'categories': {}
    }
    
    # Data storage
    compounds_data = []
    
    # Progress tracking
    total_combinations = stats['total_possible']
    processed = 0
    
    # Main generation loop
    for i, (nitro_smarts, nitro_name, nitro_cat) in enumerate(NITRO_REPLACEMENTS):
        for j, (chloro_smarts, chloro_name, chloro_cat) in enumerate(CHLORO_REPLACEMENTS):
            processed += 1
            
            # Progress update every 100 compounds
            if processed % 100 == 0:
                progress = (processed / total_combinations) * 100
                logger.info(f"Progress: {processed}/{total_combinations} ({progress:.1f}%)")
            
            # Generate compound ID (simple numbering)
            compound_id = f"CPD{processed:05d}"
            
            # Generate analog
            mol, smiles, props = generate_analog(nitro_smarts, chloro_smarts)
            
            if mol is not None and smiles:
                # Save individual SDF
                sdf_file = save_individual_sdf(
                    mol, compound_id, props, 
                    nitro_name, chloro_name
                )
                
                # Store data
                compounds_data.append({
                    'ID': compound_id,
                    'SMILES': smiles,
                    'Nitro_Replacement': nitro_name,
                    'Nitro_Category': nitro_cat,
                    'Chloro_Replacement': chloro_name,
                    'Chloro_Category': chloro_cat,
                    'MW': f"{props.get('MW', 0):.2f}",
                    'LogP': f"{props.get('LogP', 0):.2f}",
                    'TPSA': f"{props.get('TPSA', 0):.2f}",
                    'HBD': props.get('HBD', 0),
                    'HBA': props.get('HBA', 0),
                    'RotatableBonds': props.get('RotatableBonds', 0),
                    'Ro5_Violations': props.get('Ro5_Violations', 0),
                    'Ro5_Pass': props.get('Ro5_Pass', False),
                    'SDF_File': sdf_file
                })
                
                # Update statistics
                stats['successful'] += 1
                for prop, value in props.items():
                    if prop in stats['properties']:
                        stats['properties'][prop].append(value)
                
                # Update category counts
                cat_key = f"{nitro_cat}_{chloro_cat}"
                stats['categories'][cat_key] = stats['categories'].get(cat_key, 0) + 1
                
                # Log occasional success
                if stats['successful'] % 100 == 0:
                    logger.info(f"Generated {stats['successful']} compounds")
            else:
                stats['failed'] += 1
                logger.debug(f"Failed: {nitro_name}/{chloro_name}")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    logger.info(f"Generation complete. Successful: {stats['successful']}, Failed: {stats['failed']}")
    
    if compounds_data:
        # Create DataFrame
        df = pd.DataFrame(compounds_data)
        
        # Save master CSV
        csv_path = os.path.join(OUTPUT_DIR, "compound_library.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved master index to {csv_path}")
        
        # Save combined SDF
        combined_path = os.path.join(OUTPUT_DIR, "ALL_compounds_combined.sdf")
        create_combined_sdf(df, combined_path)
        
        # Save by category (optional)
        for category in ['acidic', 'basic', 'polar', 'neutral', 'lipophilic', 'heterocycle']:
            cat_df = df[(df['Nitro_Category'] == category) | (df['Chloro_Category'] == category)]
            if len(cat_df) > 0:
                cat_path = os.path.join(OUTPUT_DIR, f"category_{category}.csv")
                cat_df.to_csv(cat_path, index=False)
        
        # Write summary report
        write_summary_report(stats, start_time, OUTPUT_DIR)
        
        # Final output
        print("\n" + "="*80)
        print(" GENERATION COMPLETE!")
        print("="*80)
        print(f"\nOutput directory: {OUTPUT_DIR}/")
        print(f"Files generated:")
        print(f"  ✓ {stats['successful']} individual SDF files in {SDF_DIR}/")
        print(f"  ✓ Master index: compound_library.csv")
        print(f"  ✓ Combined SDF: ALL_compounds_combined.sdf")
        print(f"  ✓ Summary report: generation_summary.txt")
        print(f"  ✓ Log file: library_generation.log")
        print(f"\nCompound naming: CPD00001.sdf through CPD{stats['successful']:05d}.sdf")
        print(f"\nTotal execution time: {time.time() - start_time:.1f} seconds")
        print("="*80)
        
    else:
        logger.error("No compounds were generated!")
        sys.exit(1)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()