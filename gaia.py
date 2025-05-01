"""
GaiaTESS Pipeline: An enhanced pipeline for exoplanet transit detection
combining Gaia DR3 stellar parameters with TESS photometry data.

Author: [Author Names]
License: MIT
GitHub: https://github.com/exoplanet-transit-pipeline/gaiatess
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from astropy.timeseries import BoxLeastSquares
from astropy.stats import sigma_clip
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.signal import savgol_filter
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs, Observations
import lightkurve as lk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GaiaTESSPipeline:
    """Main pipeline class for combining Gaia DR3 and TESS data for exoplanet transit detection."""
    
    def __init__(self, output_dir="results", n_workers=4):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save results, by default "results"
        n_workers : int, optional
            Number of parallel workers, by default 4
        """
        self.output_dir = output_dir
        self.n_workers = n_workers
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "light_curves"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "candidates"), exist_ok=True)
        
        # Initialize container for results
        self.gaia_results = None
        self.transit_candidates = []

    def query_gaia_stars(self, ra, dec, radius, magnitude_limit=16.0, temp_range=None):
        """
        Query Gaia DR3 for stars in a specific region with optional filters.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        radius : float
            Search radius in degrees
        magnitude_limit : float, optional
            Upper G magnitude limit, by default 16.0
        temp_range : tuple, optional
            Temperature range filter (min, max) in K, by default None
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with Gaia DR3 stellar parameters
        """
        print(f"Querying Gaia DR3 catalog around RA={ra}, Dec={dec}, radius={radius} degrees...")
        
        # Construct the ADQL query
        query = f"""
        SELECT source_id, ra, dec, parallax, pmra, pmdec, radial_velocity,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
               teff_gspphot, radius_gspphot, lum_gspphot, 
               classprob_dsc_combmod_star
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        AND phot_g_mean_mag < {magnitude_limit}
        """
        
        # Add temperature filter if provided
        if temp_range is not None:
            query += f" AND teff_gspphot BETWEEN {temp_range[0]} AND {temp_range[1]}"
        
        # Execute the query
        job = Gaia.launch_job_async(query)
        results = job.get_results()
        
        # Convert to pandas DataFrame
        self.gaia_results = results.to_pandas()
        print(f"Found {len(self.gaia_results)} stars matching criteria.")
        
        return self.gaia_results
    
    def get_tess_data(self, gaia_source_id):
        """
        Get TESS light curve data for a specific Gaia source.
        
        Parameters
        ----------
        gaia_source_id : int
            Gaia DR3 source ID
        
        Returns
        -------
        lightkurve.LightCurve or None
            Stitched TESS light curve if available, otherwise None
        """
        # Get star info from our cached Gaia results
        star_info = self.gaia_results[self.gaia_results['source_id'] == gaia_source_id].iloc[0]
        
        # Create SkyCoord object for the star
        coord = SkyCoord(ra=star_info['ra']*u.degree, dec=star_info['dec']*u.degree)
        
        # Try narrow search first
        print(f"Searching for TESS data for Gaia source {gaia_source_id}...")
        search_result = lk.search_lightcurve(coord, radius=0.002*u.deg, mission='TESS')
        
        # If no result, try wider search
        if len(search_result) == 0:
            search_result = lk.search_lightcurve(coord, radius=0.005*u.deg, mission='TESS')
        
        # If still no result, try MAST catalog to find TIC ID
        if len(search_result) == 0:
            try:
                # Query the TIC catalog
                catalog_data = Catalogs.query_region(coord, radius=0.01*u.deg, catalog="TIC")
                if len(catalog_data) > 0:
                    # Get the nearest TIC object
                    tic_id = catalog_data[0]['ID']
                    # Search by TIC ID
                    search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            except Exception as e:
                print(f"Error querying MAST catalog: {e}")
                return None
        
        # If still no data found, return None
        if len(search_result) == 0:
            print(f"No TESS data found for Gaia source {gaia_source_id}")
            return None
        
        # Download all available PDCSAP light curves
        all_lcs = []
        sector_list = []
        print(f"Found data in {len(search_result)} TESS sectors")
        
        for lc_file in search_result:
            if 'SPOC' in lc_file.mission:  # Only use SPOC pipeline data
                try:
                    # Download and extract PDCSAP flux
                    lc = lc_file.download().PDCSAP_FLUX.remove_nans()
                    sector = lc_file.mission.split(' ')[1].replace('Sector', '').strip()
                    sector_list.append(sector)
                    all_lcs.append(lc)
                except Exception as e:
                    print(f"Error downloading sector {lc_file.mission}: {e}")
        
        if not all_lcs:
            print(f"Failed to download usable data for Gaia source {gaia_source_id}")
            return None
        
        # Stitch all sectors together
        try:
            stitched_lc = lk.LightCurveCollection(all_lcs).stitch()
            print(f"Successfully stitched light curve from sectors: {', '.join(sector_list)}")
            
            # Save the raw light curve to CSV
            lc_csv_path = os.path.join(self.output_dir, "light_curves", f"Gaia_{gaia_source_id}_TESS_raw.csv")
            lc_df = pd.DataFrame({
                'time': stitched_lc.time.value,
                'flux': stitched_lc.flux.value,
                'flux_err': stitched_lc.flux_err.value
            })
            lc_df.to_csv(lc_csv_path, index=False)
            
            return stitched_lc
        
        except Exception as e:
            print(f"Error stitching light curves: {e}")
            return None
    
    def process_light_curve(self, lc):
        """
        Process a light curve for transit detection.
        
        Parameters
        ----------
        lc : lightkurve.LightCurve
            Raw TESS light curve
        
        Returns
        -------
        lightkurve.LightCurve
            Processed light curve
        """
        # Normalize the light curve
        norm_lc = lc.normalize()
        
        # Remove outliers with sigma clipping
        clean_flux = sigma_clip(norm_lc.flux.value, sigma=5)
        outlier_mask = ~clean_flux.mask
        norm_lc = norm_lc[outlier_mask]
        
        # Detrend using Savitzky-Golay filter
        window_length = min(1001, len(norm_lc) // 4 * 2 + 1)  # Must be odd
        if window_length > 3:  # Need at least 3 points for filter
            trend = savgol_filter(norm_lc.flux.value, window_length, 2)
            detrended_flux = norm_lc.flux.value / trend
            detrended_lc = lk.LightCurve(
                time=norm_lc.time,
                flux=detrended_flux,
                flux_err=norm_lc.flux_err.value / trend
            )
        else:
            detrended_lc = norm_lc
        
        return detrended_lc
    
    def detect_transits(self, lc, gaia_source_id, stellar_radius=None, min_snr=7, min_transits=2):
        """
        Run Box Least Squares transit detection on a light curve.
        
        Parameters
        ----------
        lc : lightkurve.LightCurve
            Processed light curve
        gaia_source_id : int
            Gaia DR3 source ID
        stellar_radius : float, optional
            Stellar radius in solar radii, by default None
        min_snr : float, optional
            Minimum SNR for transit detection, by default 7
        min_transits : int, optional
            Minimum number of transits required, by default 2
        
        Returns
        -------
        dict or None
            Transit candidate parameters if detection successful, otherwise None
        """
        # Get time and flux arrays
        time = lc.time.value
        flux = lc.flux.value
        
        # Calculate duration range based on expected transit durations
        # For circular orbits, transit duration scales with P^(1/3)
        min_period = 0.5  # days
        max_period = min(30, (time[-1] - time[0]) / 2)  # days
        
        # Default stellar radius if not provided (Sun-like)
        if stellar_radius is None:
            stellar_radius = 1.0  # solar radii
        
        # Setup BLS model
        durations = np.linspace(0.05, 0.2, 10)  # fractional durations to test (as fraction of period)
        
        bls = BoxLeastSquares(time * u.day, flux)
        
        # Run BLS algorithm
        print(f"Running transit detection for Gaia source {gaia_source_id}...")
        try:
            results = bls.power(
                period=np.arange(min_period, max_period, 0.01),
                duration=durations,
                frequency_factor=0.8  # Use finer frequency grid
            )
            
            # Get the peak of the periodogram
            index = np.argmax(results.power)
            period = results.period[index].value
            t0 = results.transit_time[index].value
            duration = results.duration[index].value
            depth = results.depth[index]
            
            # Calculate SNR (depth / out-of-transit std)
            in_transit = bls.transit_mask(
                time * u.day,
                period * u.day,
                duration * u.day,
                t0 * u.day
            )
            oot_flux = flux[~in_transit]
            snr = depth / np.std(oot_flux)
            
            # Calculate number of observed transits
            transit_times = t0 + np.arange(-100, 100) * period
            transit_times = transit_times[(transit_times > time[0]) & (transit_times < time[-1])]
            n_transits = len(transit_times)
            
            # Check if candidate meets criteria
            if snr >= min_snr and n_transits >= min_transits:
                # Calculate planet radius in Earth radii
                if stellar_radius is not None:
                    planet_radius = stellar_radius * 109.2 * np.sqrt(depth)  # in Earth radii
                else:
                    planet_radius = None
                
                # Create transit parameters dictionary
                transit_params = {
                    'gaia_source_id': gaia_source_id,
                    'period': period,
                    't0': t0,
                    'duration': duration,
                    'depth_ppm': depth * 1e6,
                    'snr': snr,
                    'n_transits': n_transits,
                    'stellar_radius_solar': stellar_radius,
                    'planet_radius_earth': planet_radius
                }
                
                # Create diagnostic plots
                self._create_transit_plots(time, flux, transit_params, in_transit)
                
                return transit_params
            else:
                print(f"No significant transit detected (SNR={snr:.2f}, transits={n_transits})")
                return None
                
        except Exception as e:
            print(f"Error in transit detection: {e}")
            return None
    
    def _create_transit_plots(self, time, flux, transit_params, in_transit):
        """
        Create diagnostic plots for a transit candidate.
        
        Parameters
        ----------
        time : numpy.ndarray
            Time array
        flux : numpy.ndarray
            Flux array
        transit_params : dict
            Transit parameters
        in_transit : numpy.ndarray
            Boolean mask for in-transit points
        """
        gaia_id = transit_params['gaia_source_id']
        period = transit_params['period']
        t0 = transit_params['t0']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Full light curve with transit times marked
        ax = axes[0]
        ax.scatter(time, flux, s=2, alpha=0.5, color='blue')
        
        # Mark transit times
        for t in np.arange(t0, time[-1] + period, period):
            if t >= time[0] and t <= time[-1]:
                ax.axvline(t, alpha=0.3, color='red')
        
        ax.set_title(f"Gaia {gaia_id}: P={period:.2f}d, Depth={transit_params['depth_ppm']:.0f}ppm, SNR={transit_params['snr']:.1f}")
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Normalized Flux')
        
        # Plot 2: Phase-folded light curve
        ax = axes[1]
        phase = (time - t0) / period
        phase = phase % 1
        phase[phase > 0.5] -= 1
        
        # Sort by phase for cleaner plotting
        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        phase_flux = flux[sort_idx]
        
        # Plot individual points
        ax.scatter(phase, phase_flux, s=2, alpha=0.5, color='blue')
        
        # Plot binned data
        bins = np.linspace(-0.5, 0.5, 50)
        bin_indices = np.digitize(phase, bins)
        bin_means = [phase_flux[bin_indices == i].mean() for i in range(1, len(bins))]
        bin_centers = bins[:-1] + np.diff(bins) / 2
        
        ax.scatter(bin_centers, bin_means, color='red', s=20)
        
        ax.set_title('Phase-folded Light Curve')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_xlim(-0.5, 0.5)
        
        # Plot 3: Zoomed view of transit
        ax = axes[2]
        mask = (phase > -0.2) & (phase < 0.2)
        ax.scatter(phase[mask], phase_flux[mask], s=2, alpha=0.5, color='blue')
        
        # Plot binned data in zoom view
        zoom_bins = np.linspace(-0.2, 0.2, 40)
        zoom_bin_indices = np.digitize(phase, zoom_bins)
        zoom_bin_means = [phase_flux[zoom_bin_indices == i].mean() for i in range(1, len(zoom_bins))]
        zoom_bin_centers = zoom_bins[:-1] + np.diff(zoom_bins) / 2
        
        ax.scatter(zoom_bin_centers, zoom_bin_means, color='red', s=20)
        
        ax.set_title('Transit Close-up')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_xlim(-0.2, 0.2)
        
        # Add depth line
        depth = 1 - transit_params['depth_ppm'] / 1e6
        ax.axhline(depth, linestyle='--', color='green', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(self.output_dir, "candidates", f"Gaia_{gaia_id}_transit.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
    
    def process_star(self, gaia_source_id):
        """
        Process a single star through the full pipeline.
        
        Parameters
        ----------
        gaia_source_id : int
            Gaia DR3 source ID
        
        Returns
        -------
        dict or None
            Transit candidate parameters if detected, otherwise None
        """
        try:
            # Get star info
            star_info = self.gaia_results[self.gaia_results['source_id'] == gaia_source_id].iloc[0]
            stellar_radius = star_info['radius_gspphot'] if 'radius_gspphot' in star_info and not np.isnan(star_info['radius_gspphot']) else None
            
            # Get TESS data
            tess_lc = self.get_tess_data(gaia_source_id)
            if tess_lc is None:
                return None
            
            # Process light curve
            processed_lc = self.process_light_curve(tess_lc)
            
            # Save processed light curve
            lc_csv_path = os.path.join(self.output_dir, "light_curves", f"Gaia_{gaia_source_id}_TESS_processed.csv")
            lc_df = pd.DataFrame({
                'time': processed_lc.time.value,
                'flux': processed_lc.flux.value,
                'flux_err': processed_lc.flux_err.value if processed_lc.flux_err is not None else np.ones_like(processed_lc.flux.value)
            })
            lc_df.to_csv(lc_csv_path, index=False)
            
            # Detect transits
            candidate = self.detect_transits(processed_lc, gaia_source_id, stellar_radius)
            
            return candidate
            
        except Exception as e:
            print(f"Error processing Gaia source {gaia_source_id}: {e}")
            return None
    
    def run_pipeline(self, target_list=None):
        """
        Run the full pipeline on a list of Gaia sources.
        
        Parameters
        ----------
        target_list : list, optional
            List of Gaia source IDs to process, by default None.
            If None, all stars in self.gaia_results will be processed.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with transit candidates
        """
        if self.gaia_results is None:
            print("No Gaia results available. Run query_gaia_stars first.")
            return None
        
        if target_list is None:
            target_list = self.gaia_results['source_id'].values
        
        print(f"Processing {len(target_list)} stars with {self.n_workers} parallel workers...")
        
        # Process stars in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(executor.map(self.process_star, target_list), total=len(target_list)))
        
        # Filter out None results and create DataFrame
        candidates = [r for r in results if r is not None]
        self.transit_candidates = candidates
        
        if candidates:
            # Save candidates to CSV
            candidates_df = pd.DataFrame(candidates)
            candidates_df.to_csv(os.path.join(self.output_dir, "transit_candidates.csv"), index=False)
            print(f"Found {len(candidates)} transit candidates.")
            return candidates_df
        else:
            print("No transit candidates found.")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GaiaTESSPipeline(output_dir="results", n_workers=4)
    
    # Query Gaia for stars in a specific region
    stars = pipeline.query_gaia_stars(
        ra=88.793, 
        dec=0.871, 
        radius=0.5,
        magnitude_limit=15.0,
        temp_range=(4000, 7000)  # K-G stars
    )
    
    # Run pipeline on all stars
    candidates = pipeline.run_pipeline()
    
    # Print results
    if not candidates.empty:
        print("\nTransit Candidates:")
        print(candidates[['gaia_source_id', 'period', 'depth_ppm', 'snr', 'n_transits', 'planet_radius_earth']])
