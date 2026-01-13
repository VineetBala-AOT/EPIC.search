import { useEffect, useState } from "react";
import { Cancel, Search as SearchIcon, Settings, Psychology, AutoAwesomeTwoTone, Summarize, FindInPage } from "@mui/icons-material";
import {
  Box,
  Grid,
  Typography,
  Paper,
  Tooltip, 
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Stack
} from "@mui/material";
import { createFileRoute, Navigate, useSearch } from "@tanstack/react-router";
import { FilterSidebar } from '../../components/App/Filters/FilterSideBar';
import { BCDesignTokens } from "epic.theme";
import { useRoles } from "@/contexts/AuthContext";
import { getStoredSearchStrategy, setStoredSearchStrategy, RankingConfig, getStoredRankingConfig, scoreSettings, resultsSettings, getStoredSearchMode, setStoredSearchMode } from "@/utils/searchConfig";
import { SearchMode } from "@/hooks/useSearch";
import { InputBase } from "@mui/material";
import { useSearchQuery, SearchStrategy, SearchRequest } from "@/hooks/useSearch";
import { SearchResponse } from "@/models/Search";
import { LocationControl } from "@/components/Location";
import { FeedbackControl } from "@/components/App/Feedback";
import SearchConfigModal from "@/components/App/Search/SearchConfigModal";
import SearchLandingAccordion from "@/components/App/Search/SearchLanding";
import SearchSkelton from "@/components/App/Search/SearchSkelton";
import SearchResult from "@/components/App/Search/SearchResult";
import { useDocumentTypeMappings } from "@/hooks/useDocumentTypeMappings";
import { useProjects } from "@/hooks/useProjects";
import { PageLoader } from "@/components/PageLoader";

const SIDEBAR_WIDTH = 320;
const COLLAPSED_WIDTH = 56;

export const Route = createFileRoute("/_authenticated/search")({
  component: Search,
});

function Search() {
  localStorage.removeItem("authReturnPath");
  const { data: rawProjects, isLoading: loadingProjects, isError: projectError } = useProjects();
  const {
    data: rawDocTypes,
    isLoading: loadingDocTypes,
    isError: docTypeError,
  } = useDocumentTypeMappings();

  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      const raw = localStorage.getItem("collapsed");
      return raw ? JSON.parse(raw) : false; // default open
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("collapsed", JSON.stringify(collapsed));
    } catch {}
  }, [collapsed]);

  const [searchText, setSearchText] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [selectedProjects, setSelectedProjects] = useState<string[]>([]);
  const [selectedDocTypes, setSelectedDocTypes] = useState<string[]>([]);
  const [searchMode, setSearchMode] = useState<SearchMode>(getStoredSearchMode());
  const [modeMenuAnchor, setModeMenuAnchor] = useState<null | HTMLElement>(null);
  const [feedbackSessionId, setFeedbackSessionId] = useState<string | null>(null);
  const [rankingConfig, setRankingConfig] = useState<RankingConfig>(getStoredRankingConfig());
  const [configModalOpen, setConfigModalOpen] = useState(false);

  const { isAdmin, isViewer } = useRoles();

  const onSubmitSearchWithQuery = (query: string) => {
    const searchRequest: SearchRequest = {
      query,
      searchStrategy: isViewer ? "HYBRID_PARALLEL" : searchStrategy,
      ...(selectedProjects.length > 0 && { projectIds: selectedProjects }),
      ...(selectedDocTypes.length > 0 && { documentTypeIds: selectedDocTypes }),
      ...(rankingConfig.useCustomRanking && {
        ranking: {
          minScore: scoreSettings[isViewer ? "FLEXIBLE" : rankingConfig.scoreSetting],
          topN: resultsSettings[isViewer ? "MEDIUM" : rankingConfig.resultsSetting],
        },
      }),
      mode: isViewer ? "agent" : searchMode,
    };
    doSearch(searchRequest);

    // Update the URL with query parameter
    const url = new URL(window.location.href);
    url.searchParams.set("keywords", query);
    window.history.replaceState(null, "", url.toString());
  };

  const handleProjectToggle = (id: string) => {
    setSelectedProjects((prev) =>
      prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id]
    );
  };

  const handleDocTypeToggle = (id: string) => {
    setSelectedDocTypes((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  const handleClearAll = () => {
    setSelectedProjects([]);
    setSelectedDocTypes([]);
  };

  const handleSaveSearchStrategy = (strategy: SearchStrategy | undefined, newRankingConfig: RankingConfig) => {
    setStoredSearchStrategy(strategy);
    setSearchStrategy(strategy);
    setRankingConfig(newRankingConfig);
  };

  const handleModeChange = (newMode: SearchMode) => {
    setSearchMode(newMode);
    setStoredSearchMode(newMode);
    setModeMenuAnchor(null);
  };

  const handleModeMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setModeMenuAnchor(event.currentTarget);
  };

  const handleModeMenuClose = () => {
    setModeMenuAnchor(null);
  };

  const getModeLabel = (mode: SearchMode): string => {
    switch (mode) {
      case 'rag':
        return 'RAG Search';
      case 'ai':
        return 'AI Search';
      case 'agent':
        return 'Agent Search';
      case 'summary':
        return 'Summary Search';
      default:
        return 'RAG Search';
    }
  };

  const getModeDescription = (mode: SearchMode): string => {
    switch (mode) {
      case 'rag':
        return 'Pure vector search, fastest';
      case 'ai':
        return 'Use NLP to interpret your query and summarize';
      case 'agent':
        return 'Full AI processing with agent capabilities';
      case 'summary':
        return 'Use RAG search and summarize';
      default:
        return 'Pure vector search, fastest';
    }
  };

  const getModeIcon = (mode: SearchMode) => {
    switch (mode) {
      case 'rag':
        return <FindInPage sx={{ fontSize: 24 }} />;
      case 'ai':
        return <Psychology sx={{ fontSize: 24 }} />;
      case 'agent':
        return <AutoAwesomeTwoTone sx={{ fontSize: 24, color: BCDesignTokens.themeBlue100 }} />;
      case 'summary':
        return <Summarize sx={{ fontSize: 24 }} />;
      default:
        return <SearchIcon sx={{ fontSize: 24 }} />;
    }
  };

  const [searchStrategy, setSearchStrategy] = useState<SearchStrategy | undefined>(() => {
    const stored = getStoredSearchStrategy();

    // Viewer: always default to HYBRID_PARALLEL
    if (isViewer) return "HYBRID_PARALLEL";

    // Admin: use stored value or undefined
    return stored ?? "HYBRID_PARALLEL";
  });

  const onSuccess = (data: SearchResponse) => {
    try {
      // Validate the response structure
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid response: Expected object');
      }
      
      if (!data.result) {
        throw new Error('Invalid response: Missing result property');
      }
      
      // Ensure documents array exists or create empty array
      if (!data.result.documents && !data.result.document_chunks) {
        console.warn('Response has no documents or document_chunks, creating empty result');
        data.result.documents = [];
      }
      
      setSearchResults(data);

      // Store feedback session ID if it exists
      if ('feedback_session_id' in data && typeof data.feedback_session_id === 'string') {
        setFeedbackSessionId(data.feedback_session_id);
      } else {
        setFeedbackSessionId(null);
      }
    } catch (error) {
      console.error('Error processing search response:', error);
      // Treat malformed response as an error
      onError(error);
    }
  };

  const onError = (error: any) => {
    console.error('Search error:', error);
    setSearchResults(null);
    
    // You could also show a user-friendly error message here
    // For example, using a toast notification or error state
  };

  const {
    mutate: doSearch,
    isPending,
    error,
    isSuccess,
    reset
  } = useSearchQuery(onSuccess, onError);

  const onSubmitSearch = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchText.trim()) return;
    onSubmitSearchWithQuery(searchText.trim());
  };

  const searchParams = useSearch({ from: "/_authenticated/search" });
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    // Support both possible param names
    const initialQuery = params.get("keywords")?.trim();

    if (initialQuery) {
      setSearchText(initialQuery);

      // ensure URL stores it consistently as ?q=
      const url = new URL(window.location.href);
      url.searchParams.set("q", initialQuery);
      window.history.replaceState(null, "", url.toString());

      onSubmitSearchWithQuery(initialQuery);
    }
  }, [searchParams]);

  useEffect(() => {
    if (!searchText) {
      setSearchResults(null);
      reset();
    }
  }, [searchText]);


  if (loadingProjects || loadingDocTypes) return <PageLoader />;
  if (projectError || docTypeError) return <Navigate to="/error" />;

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "row",
        minHeight: "100vh",
        width: "100%",
        paddingTop: "60px",
        maxWidth: "1448px",
      }}
    >
      {/* Sidebar */}
      <Box
        sx={{
          width: collapsed ? `${COLLAPSED_WIDTH}px` : `${SIDEBAR_WIDTH}px`,
          minWidth: collapsed ? `${COLLAPSED_WIDTH}px` : `${SIDEBAR_WIDTH}px`,
          backgroundColor: `${BCDesignTokens.themeBlue10}`,
          display: "flex",
          flexDirection: "column",
          borderRight: "1px solid rgba(0,0,0,0.12)",
        }}
      >
        <FilterSidebar
          rawProjects={rawProjects ?? []}
          rawDocTypes={rawDocTypes}
          selectedProjects={selectedProjects}
          selectedDocTypes={selectedDocTypes}
          onProjectToggle={handleProjectToggle}
          onDocTypeToggle={handleDocTypeToggle}
          onClearAll={handleClearAll}
          collapsed={collapsed}
          onCollapseToggle={() => setCollapsed((s) => !s)}
        />
      </Box>

      {/* Main Content */}
      <Box
        sx={{
          flexGrow: 1,
          p: 3,
          overflow: "auto",
        }}
      >
        <Paper elevation={0} sx={{ p: 3 }}>
          <Typography
            variant="h2"
            sx={{
              mb: 0,
              textAlign: "left",
            }}
          >
            Document Search
          </Typography>
          <Typography
            variant="body2"
            sx={{ mt: 1, mb: 4, textAlign: "left", color: BCDesignTokens.themeGray80, fontSize: "16px" }}
          >
            Search for documents by entering a keyword or phrase below.
          </Typography>
          <Stack spacing={1.5}>
            <Paper
              component="form"
              sx={{
                borderRadius: "8px",
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
              {isAdmin &&
                <Tooltip title={`${getModeLabel(searchMode)}: ${getModeDescription(searchMode)}`}>
                  <IconButton
                    onClick={handleModeMenuOpen}
                    sx={{
                      mr: 1,
                      borderRadius: "12px",
                      minWidth: "48px",
                      height: "48px",
                      color: BCDesignTokens.iconsColorSuccess,
                      "&:hover": {
                        backgroundColor: BCDesignTokens.supportSurfaceColorSuccess,
                      },
                    }}
                  >
                    {getModeIcon(searchMode)}
                  </IconButton>
                </Tooltip>
              }
              
              <Menu
                anchorEl={modeMenuAnchor}
                open={Boolean(modeMenuAnchor)}
                onClose={handleModeMenuClose}
                anchorOrigin={{
                  vertical: 'bottom',
                  horizontal: 'left',
                }}
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'left',
                }}
                PaperProps={{
                  sx: {
                    borderRadius: '12px',
                    boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
                  }
                }}
              >
                <MenuItem 
                  onClick={() => handleModeChange('rag')}
                  selected={searchMode === 'rag'}
                  sx={{
                    px: 2,
                    py: 1.5,
                    backgroundColor: searchMode === 'rag' ? BCDesignTokens.themeBlue10 : 'transparent',
                    '&:hover': {
                      backgroundColor: BCDesignTokens.themeBlue10,
                    }
                  }}
                >
                  <ListItemIcon sx={{ minWidth: '36px' }}>
                    <FindInPage sx={{ fontSize: 20, color: BCDesignTokens.themePrimaryBlue }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="RAG Search"
                    secondary="Pure vector search, fastest"
                    secondaryTypographyProps={{
                      sx: { fontSize: '0.75rem', color: BCDesignTokens.themeGray60 }
                    }}
                  />
                </MenuItem>
                
                <MenuItem 
                  onClick={() => handleModeChange('summary')}
                  selected={searchMode === 'summary'}
                  sx={{
                    px: 2,
                    py: 1.5,
                    backgroundColor: searchMode === 'summary' ? BCDesignTokens.themeBlue10 : 'transparent',
                    '&:hover': {
                      backgroundColor: BCDesignTokens.themeBlue10,
                    }
                  }}
                >
                  <ListItemIcon sx={{ minWidth: '36px' }}>
                    <Summarize sx={{ fontSize: 20, color: BCDesignTokens.themePrimaryBlue }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Summary Search"
                    secondary="Use RAG search and summarize"
                    secondaryTypographyProps={{
                      sx: { fontSize: '0.75rem', color: BCDesignTokens.themeGray60 }
                    }}
                  />
                </MenuItem>
                
                <MenuItem 
                  onClick={() => handleModeChange('ai')}
                  selected={searchMode === 'ai'}
                  sx={{
                    px: 2,
                    py: 1.5,
                    backgroundColor: searchMode === 'ai' ? BCDesignTokens.themeBlue10 : 'transparent',
                    '&:hover': {
                      backgroundColor: BCDesignTokens.themeBlue10,
                    }
                  }}
                >
                  <ListItemIcon sx={{ minWidth: '36px' }}>
                    <Psychology sx={{ fontSize: 20, color: BCDesignTokens.themePrimaryBlue }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="AI Search"
                    secondary="Use NLP to interpret your query and summarize"
                    secondaryTypographyProps={{
                      sx: { fontSize: '0.75rem', color: BCDesignTokens.themeGray60 }
                    }}
                  />
                </MenuItem>
                
                <MenuItem 
                  onClick={() => handleModeChange('agent')}
                  selected={searchMode === 'agent'}
                  sx={{
                    px: 2,
                    py: 1.5,
                    backgroundColor: searchMode === 'agent' ? BCDesignTokens.themeBlue10 : 'transparent',
                    '&:hover': {
                      backgroundColor: BCDesignTokens.themeBlue10,
                    }
                  }}
                >
                  <ListItemIcon sx={{ minWidth: '36px' }}>
                    <AutoAwesomeTwoTone sx={{ fontSize: 20, color: BCDesignTokens.themeBlue100 }} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Agent Search"
                    secondary="Full AI processing with agent capabilities"
                    secondaryTypographyProps={{
                      sx: { fontSize: '0.75rem', color: BCDesignTokens.themeGray60 }
                    }}
                  />
                </MenuItem>
              </Menu>
              <InputBase
                sx={{ ml: 1, flex: 1, height: 64 }}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder={searchMode !== 'rag' ? "Ask AI about documents..." : "Search documents..."}
                inputProps={{ "aria-label": "search text" }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    onSubmitSearch(e);
                  }
                }}
              />
              {isAdmin &&
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
              }
              {isPending ? (
                <Tooltip title="Cancel search">
                  <IconButton
                    type="button"
                    sx={{ p: "10px" }}
                    aria-label="cancel search"
                    size="large"
                    onClick={() => {
                      reset();
                      setSearchResults(null);
                      setSearchText("");
                    }}
                  >
                    <Cancel sx={{ fontSize: 30, color: '#d32f2f' }} />
                  </IconButton>
                </Tooltip>
              ) : searchText ? (
                <>
                  <Tooltip title="Search">
                    <IconButton
                      type="button"
                      sx={{ p: "10px" }}
                      aria-label="search"
                      size="large"
                      onClick={onSubmitSearch}
                    >
                      <SearchIcon sx={{ fontSize: 30 }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Clear search">
                    <IconButton
                      type="button"
                      sx={{ p: "10px" }}
                      aria-label="clear search"
                      size="large"
                      onClick={() => setSearchText("")}
                    >
                      <Cancel sx={{ fontSize: 30 }} />
                    </IconButton>
                  </Tooltip>
                </>
              ) : (
                <Tooltip title="Search">
                  <IconButton
                    type="button"
                    sx={{ p: "10px" }}
                    aria-label="search"
                    size="large"
                    onClick={onSubmitSearch}
                  >
                    <SearchIcon sx={{ fontSize: 30 }} />
                  </IconButton>
                </Tooltip>
              )}
            </Paper>
            <Grid
              container
              alignItems="center"
              justifyContent="space-between"
              wrap="nowrap"
              sx={{ gap: 1 }}
            >
              <Grid item sx={{ flex: 1, minWidth: 0 }}>
                <FeedbackControl sessionId={feedbackSessionId ?? undefined} />
              </Grid>

              <Grid item sx={{ flexShrink: 0, minWidth: 0 }}>
                <LocationControl />
              </Grid>
            </Grid>
          </Stack>
          {isAdmin &&
            <SearchConfigModal
              open={configModalOpen}
              onClose={() => setConfigModalOpen(false)}
              currentStrategy={searchStrategy}
              onSave={handleSaveSearchStrategy}
            />
          }
          <Box
            sx={{
              mt: 2,
              display: "flex",
              flexDirection: "column",
              alignContent: "left",
              alignItems: "left",
            }}
          >
            {!searchResults && !isPending && !error && <SearchLandingAccordion />}
            {isPending && <SearchSkelton />}
            {error && (
              <Box 
                sx={{ 
                  textAlign: 'center', 
                  mt: 4, 
                  p: 3, 
                  borderRadius: 2, 
                  backgroundColor: 'error.light',
                  color: 'error.contrastText'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  Search Error
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {error.message || 'An error occurred while searching. Please try again.'}
                </Typography>
                <Typography variant="body2" sx={{ mt: 2, opacity: 0.8 }}>
                  If the problem persists, please contact support.
                </Typography>
              </Box>
            )}
            {isSuccess && searchResults?.result && (
              <SearchResult 
                searchResults={searchResults} 
                searchText={searchText} 
                searchMode={searchMode}
              />
            )}
          </Box>
        </Paper>
      </Box>
    </Box>
  );
}
