import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  RadioGroup,
  FormControlLabel,
  Radio,
  Typography,
  Box,
  IconButton,
  Tooltip,
  Divider,
  Checkbox,
  FormControl,
  FormLabel,
} from "@mui/material";
import { Close, HelpOutline } from "@mui/icons-material";
import { BCDesignTokens } from "epic.theme";
import { useState } from "react";
import { SearchStrategy } from "@/hooks/useSearch";
import { useSearchStrategies } from "@/hooks/useSearchStrategies";
import { 
  RankingConfig,
  getStoredRankingConfig,
  setStoredRankingConfig
} from "@/utils/searchConfig";

interface SearchConfigModalProps {
  open: boolean;
  onClose: () => void;
  currentStrategy: SearchStrategy | undefined;
  onSave: (strategy: SearchStrategy | undefined, rankingConfig: RankingConfig) => void;
}

const SearchConfigModal = ({ 
  open, 
  onClose, 
  currentStrategy, 
  onSave 
}: SearchConfigModalProps) => {
  const [selectedStrategy, setSelectedStrategy] = useState<SearchStrategy | undefined>(
    currentStrategy ?? "HYBRID_PARALLEL"
  );
  const [rankingConfig, setRankingConfig] = useState<RankingConfig>(getStoredRankingConfig());
  const [showHelp, setShowHelp] = useState<string | null>(null);
  
  const { data: searchStrategyOptions, isLoading: strategiesLoading, isError: strategiesError } = useSearchStrategies(true);
  
  const handleSave = () => {
    onSave(selectedStrategy, rankingConfig);
    setStoredRankingConfig(rankingConfig);
    onClose();
  };

  const handleCancel = () => {
    setSelectedStrategy(currentStrategy);
    setRankingConfig(getStoredRankingConfig());
    setShowHelp(null);
    onClose();
  };

  const toggleHelp = (value: string) => {
    setShowHelp(showHelp === value ? null : value);
  };

  return (
    <Dialog
      open={open}
      onClose={handleCancel}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
          maxHeight: "80vh",
        },
      }}
    >
      <DialogTitle
        component="div"
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: `1px solid ${BCDesignTokens.surfaceColorBorderDefault}`,
          pb: 2,
        }}
      >
        <Typography variant="h5" color={BCDesignTokens.themePrimaryBlue}>
          Search Configuration
        </Typography>
        <IconButton onClick={handleCancel} size="small">
          <Close />
        </IconButton>
      </DialogTitle>
      
      <DialogContent sx={{ py: 3 }}>
        <Typography variant="body1" sx={{ mb: 3, color: BCDesignTokens.themeGray80 }}>
          Configure your search strategy to optimize results for different types of queries.
        </Typography>

        <Typography variant="h6" sx={{ mb: 2, color: BCDesignTokens.themePrimaryBlue }}>
          Search Strategy
        </Typography>

        {strategiesLoading && (
          <Typography>Loading search strategies...</Typography>
        )}
        
        {strategiesError && (
          <Typography color="error">Failed to load search strategies. Using defaults.</Typography>
        )}

        {!strategiesLoading && searchStrategyOptions && (
          <RadioGroup
            value={selectedStrategy || ""}
            onChange={(e) => {
              const value = e.target.value;
              setSelectedStrategy(value === "" ? undefined : value as SearchStrategy);
            }}
          >
            {searchStrategyOptions.map((option) => {
              const radioValue = option.value === undefined ? "" : option.value;
              return (
                <Box key={option.value || "default"} sx={{ mb: 2 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <FormControlLabel
                      value={radioValue}
                      control={<Radio />}
                      label={option.label}
                      sx={{ flex: 1 }}
                    />
                    <Tooltip title="Click for more information">
                      <IconButton
                        size="small"
                      onClick={() => toggleHelp(option.value || "default")}
                      sx={{ 
                        color: BCDesignTokens.themeBlue70,
                        "&:hover": { color: BCDesignTokens.themeBlue90 }
                      }}
                    >
                      <HelpOutline fontSize="small" />
                    </IconButton>
                  </Tooltip>
              </Box>
              
              {showHelp === (option.value || "default") && (
                <Box
                  sx={{
                    ml: 4,
                    p: 2,
                    mt: 1,
                    backgroundColor: BCDesignTokens.themeBlue10,
                    borderRadius: 1,
                    border: `1px solid ${BCDesignTokens.themeBlue30}`,
                  }}
                >
                  <Typography variant="body2" color={BCDesignTokens.themeGray80}>
                    {option.description}
                  </Typography>
                </Box>
              )}
            </Box>
          );
            })}
          </RadioGroup>
        )}

        <Divider sx={{ my: 3 }} />

        <Typography variant="h6" sx={{ mb: 2, color: BCDesignTokens.themePrimaryBlue }}>
          Ranking Configuration
        </Typography>

        <FormControlLabel
          control={
            <Checkbox
              checked={rankingConfig.useCustomRanking}
              onChange={(e) => setRankingConfig(prev => ({ ...prev, useCustomRanking: e.target.checked }))}
              color="primary"
            />
          }
          label="Use custom ranking settings"
          sx={{ mb: 2 }}
        />

        {rankingConfig.useCustomRanking && (
          <Box sx={{ ml: 4, mb: 3 }}>
            <FormControl component="fieldset" sx={{ mb: 3, width: '100%' }}>
              <FormLabel component="legend" sx={{ mb: 1, color: BCDesignTokens.themeGray80, fontWeight: 600 }}>
                Result Relevance Threshold
              </FormLabel>
              <RadioGroup
                value={rankingConfig.scoreSetting}
                onChange={(e) => setRankingConfig(prev => ({ 
                  ...prev, 
                  scoreSetting: e.target.value as RankingConfig['scoreSetting'] 
                }))}
              >
                <FormControlLabel value="STRICT" control={<Radio />} label="Strict - Only highly relevant results" />
                <FormControlLabel value="MODERATE" control={<Radio />} label="Moderate - Good balance of relevance" />
                <FormControlLabel value="FLEXIBLE" control={<Radio />} label="Flexible - Include more diverse results" />
                <FormControlLabel value="VERY_FLEXIBLE" control={<Radio />} label="Very Flexible - Cast a wide net" />
              </RadioGroup>
            </FormControl>

            <FormControl component="fieldset" sx={{ width: '100%' }}>
              <FormLabel component="legend" sx={{ mb: 1, color: BCDesignTokens.themeGray80, fontWeight: 600 }}>
                Number of Results
              </FormLabel>
              <RadioGroup
                value={rankingConfig.resultsSetting}
                onChange={(e) => setRankingConfig(prev => ({ 
                  ...prev, 
                  resultsSetting: e.target.value as RankingConfig['resultsSetting'] 
                }))}
              >
                <FormControlLabel value="FEW" control={<Radio />} label="Few - 5 results (fastest)" />
                <FormControlLabel value="MEDIUM" control={<Radio />} label="Medium - 15 results (balanced)" />
                <FormControlLabel value="MANY" control={<Radio />} label="Many - 50 results (comprehensive)" />
                <FormControlLabel value="MAXIMUM" control={<Radio />} label="Maximum - 100 results (exhaustive)" />
              </RadioGroup>
            </FormControl>
          </Box>
        )}

        <Divider sx={{ my: 3 }} />

        <Typography variant="body2" sx={{ color: BCDesignTokens.themeGray70, fontStyle: "italic" }}>
          Your search strategy preference will be saved locally and applied to all future searches.
        </Typography>
      </DialogContent>

      <DialogActions
        sx={{
          borderTop: `1px solid ${BCDesignTokens.surfaceColorBorderDefault}`,
          p: 2,
          gap: 1,
        }}
      >
        <Button
          onClick={handleCancel}
          variant="outlined"
          sx={{
            borderColor: BCDesignTokens.themeGray40,
            color: BCDesignTokens.themeGray80,
            "&:hover": {
              borderColor: BCDesignTokens.themeGray60,
              backgroundColor: BCDesignTokens.themeGray10,
            },
          }}
        >
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          sx={{
            backgroundColor: BCDesignTokens.themePrimaryBlue,
            "&:hover": {
              backgroundColor: BCDesignTokens.themeBlue80,
            },
          }}
        >
          Save Settings
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SearchConfigModal;
