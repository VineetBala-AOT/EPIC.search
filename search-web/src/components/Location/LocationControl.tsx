import React from 'react';
import { Box, Typography, Switch, CircularProgress, IconButton, Tooltip, Alert } from '@mui/material';
import { LocationOn, Refresh, LocationOff } from '@mui/icons-material';
import { useLocation } from '@/contexts/LocationContext';

const LocationControl: React.FC = () => {
  const {
    locationData,
    isLoading,
    error,
    isLocationEnabled,
    requestLocation,
    clearLocation,
    setLocationEnabled,
    refreshLocation,
  } = useLocation();

  const handleToggle = async () => {
    if (!isLocationEnabled) {
      setLocationEnabled(true);
      await requestLocation(); // force prompt on user click
    } else {
      setLocationEnabled(false);
      clearLocation();
    }
  };

  const getLocationDisplay = () => {
    if (!locationData) return null;
    if (locationData.city && locationData.region) return `${locationData.city}, ${locationData.region}`;
    return `${locationData.latitude.toFixed(2)}, ${locationData.longitude.toFixed(2)}`;
  };

  return (
    <Box display="flex" flexDirection="column" gap={1}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          px: 1,
          py: 0.5,
          borderRadius: "8px",
          cursor: "pointer",
          minHeight: "45px",
          maxWidth: '400px',
          backgroundColor: isLocationEnabled ? "#F1F8FE" : "#F3F4F6",
          transition: "background-color 0.2s ease",
          "&:hover": {
            backgroundColor: isLocationEnabled ? "#E6F2FD" : "#E5E7EB",
          },
        }}
        onClick={handleToggle}
      >
        {isLocationEnabled ? <LocationOn sx={{ fontSize: 20, color: "#013366" }} />
        : <LocationOff sx={{ fontSize: 20, color: "#6B7280" }} />}
        <Box sx={{ display: "flex", flexDirection: "column", flex: 1 }}>
          <Typography
            sx={{
              fontSize: "0.875rem", // text-sm
              fontWeight: 400,
              color: isLocationEnabled ? "#013366" : "#374151",
              lineHeight: 1.3,
            }}
          >
            Use my location for nearby results
          </Typography>
          <Typography
            sx={{
              fontSize: "0.75rem", // text-xs
              fontWeight: 400,
              color: "#6B7280",
            }}
          >
            Your location is only used for search and not stored
          </Typography>
        </Box>
        <Switch checked={isLocationEnabled} onChange={handleToggle} color="primary" onClick={e => e.stopPropagation()} />
      </Box>

      {isLocationEnabled && (
        <Box display="flex" alignItems="center" gap={1}>
          {isLoading && <CircularProgress size={16} />}
          {locationData && <Typography variant="body2" color="text.secondary">{getLocationDisplay()}</Typography>}
          {locationData && (
            <Tooltip title="Refresh location">
              <IconButton size="small" onClick={refreshLocation} disabled={isLoading}><Refresh /></IconButton>
            </Tooltip>
          )}
        </Box>
      )}

      {error && <Alert severity="error">{error}</Alert>}
    </Box>
  );
};

export default LocationControl;
