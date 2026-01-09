import React from "react";
import { Box, Typography, Link } from "@mui/material";
import { BCDesignTokens } from "epic.theme";

const Unauthorized: React.FC = React.memo(() => {
  return (
    <Box
      p={3}
      pt={BCDesignTokens.layoutPaddingSmall}
      mt={15}
    >
      <Box
        sx={{
          height: "5px",
          width: "50px",
          backgroundColor: BCDesignTokens.themeGold100,
          ml: BCDesignTokens.layoutMarginXxxlarge
        }}
      />
      <Typography
        variant="h2"
        sx={{
          ml: BCDesignTokens.layoutMarginXxxlarge,
          width: "100%",
        }}
      >
        Need Access to AI Search?
      </Typography>
      <Typography
        variant="h6"
        gutterBottom
        fontWeight={400}
        sx={{
          ml: BCDesignTokens.layoutMarginXxxlarge,
          mt: BCDesignTokens.layoutMarginMedium,
          width: "100%",
        }}
      >
        It appears you've arrived at AI Search without proper access.
      </Typography>
      <Typography
        variant="h6"
        gutterBottom
        fontWeight={400}
        sx={{
          ml: BCDesignTokens.layoutMarginXxxlarge,
          mt: BCDesignTokens.layoutMarginMedium,
          width: "100%",
        }}
      >
        If you believe you should have access to AI Search, please
        contact the Environmental Assessment Office at
        <Link
          href="mailto:EAO.ManagementPlanSupport@gov.bc.ca"
          sx={{ ml: BCDesignTokens.layoutMarginXsmall }}
        >
          EAO.ManagementPlanSupport@gov.bc.ca.
        </Link>
      </Typography>
    </Box>
  );
});

export default Unauthorized;
