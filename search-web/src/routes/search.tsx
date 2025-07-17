import { Cancel, Search as SearchIcon, Settings } from "@mui/icons-material";
import { Box, Container, Typography, Tooltip, IconButton } from "@mui/material";
import { InputBase } from "@mui/material";
import { Paper } from "@mui/material";
import { createFileRoute } from "@tanstack/react-router";
import { BCDesignTokens } from "epic.theme";
import { useEffect, useState } from "react";
import { useSearchQuery, SearchStrategy, SearchRequest } from "@/hooks/useSearch";
import { SearchResponse } from "@/models/Search";
import SearchSkelton from "@/components/App/Search/SearchSkelton";
import SearchResult from "@/components/App/Search/SearchResult";
import SearchLanding from "@/components/App/Search/SearchLanding";
import SearchConfigModal from "@/components/App/Search/SearchConfigModal";
import { getStoredSearchStrategy, setStoredSearchStrategy } from "@/utils/searchConfig";
export const Route = createFileRoute("/search")({
  component: Search,
});

function Search() {
  const [searchText, setSearchText] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(
    null
  );
  const [configModalOpen, setConfigModalOpen] = useState(false);
  const [searchStrategy, setSearchStrategy] = useState<SearchStrategy | undefined>(
    getStoredSearchStrategy()
  );

  const onSuccess = (data: SearchResponse) => {
    setSearchResults(data);
  };

  const onError = (error: any) => {
    console.error(error);
  };

  useEffect(() => {
    if (!searchText) {
      setSearchResults(null);
      reset();
    }
  }, [searchText]);

  const {
    mutate: doSearch,
    isPending,
    error,
    isSuccess,
    reset
  } = useSearchQuery(onSuccess, onError);

  const onSubmitSearch = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const searchRequest: SearchRequest = {
      question: searchText,
      ...(searchStrategy && { searchStrategy })
    };
    doSearch(searchRequest);
  };

  const handleSaveSearchStrategy = (strategy: SearchStrategy | undefined) => {
    setStoredSearchStrategy(strategy);
    setSearchStrategy(strategy);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 1 }}>
      <Typography
        variant="h2"
        sx={{
          mb: 0,
          mt: 2,
          textAlign: "center",
          color: BCDesignTokens.themePrimaryBlue,
        }}
      >
        Document Search
      </Typography>
      <Typography
        variant="body1"
        sx={{ mb: 4, textAlign: "center", color: BCDesignTokens.themeBlue90 }}
      >
        Search for documents by entering a keyword or phrase below.
      </Typography>
      <Paper
        component="form"
        sx={{
          borderRadius: "16px",
          display: "flex",
          alignItems: "center",
          padding: "8px 16px",
          border: "1px solid",
          borderColor: BCDesignTokens.themeBlue10,
          boxShadow: "0px 2px 6px -2px rgb(0 0 0 / 33%)",
          "&:hover": {
            boxShadow: "0px 2px 18px 0px rgb(85 149 217 / 36%)",
          },
          "&:focus-within": {
            boxShadow: "0px 2px 18px 0px rgb(85 149 217 / 36%)",
          },
        }}
      >
        <InputBase
          sx={{ ml: 1, flex: 1, height: 64 }}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          placeholder="Search text..."
          inputProps={{ "aria-label": "search text" }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              onSubmitSearch(e);
            }
          }}
        />
        <Tooltip title="Search Configuration">
          <IconButton
            type="button"
            sx={{ p: "10px" }}
            aria-label="search configuration"
            size="large"
            onClick={() => setConfigModalOpen(true)}
          >
            <Settings sx={{ fontSize: 24, color: BCDesignTokens.themeGray60 }} />
          </IconButton>
        </Tooltip>
        {searchText && (
          <IconButton
            type="button"
            sx={{ p: "10px" }}
            aria-label="clear search"
            size="large"
            onClick={() => setSearchText("")}
          >
            <Cancel sx={{ fontSize: 30 }} />
          </IconButton>
        )}
        {!searchText && (
          <IconButton
            type="button"
            sx={{ p: "10px" }}
            aria-label="search"
            size="large"
            onClick={onSubmitSearch}
          >
            <SearchIcon sx={{ fontSize: 30 }} />
          </IconButton>
        )}
      </Paper>

      <SearchConfigModal
        open={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
        currentStrategy={searchStrategy}
        onSave={handleSaveSearchStrategy}
      />

      <Box
        sx={{
          mt: 2,
          display: "flex",
          flexDirection: "column",
          alignContent: "center",
          alignItems: "center",
        }}
      >
        {!searchResults && !isPending && !error && <SearchLanding />}
        {isPending && <SearchSkelton />}
        {error && <Typography>Error: {error.message}</Typography>}
        {isSuccess && searchResults?.result && (
          <SearchResult 
            searchResults={searchResults} 
            searchText={searchText} 
            searchStrategy={searchStrategy}
          />
        )}
      </Box>
    </Container>
  );
}
