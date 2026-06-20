#include <utility>
#include "state.hpp"
#include "alpha_beta.hpp"

/*============================================================
 * AlphaBeta — eval_ctx
 *
 * Negamax with Alpha-Beta pruning. Caller manages memory.
 *============================================================*/
int AlphaBeta::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const ABParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if (state->game_state == WIN){
        return M_MAX + ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop with Alpha-Beta Pruning === */
    int best_score = M_MAX; 

    for(auto& action : state->legal_actions){
        auto *next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int raw_score;
        // 如果是同一個玩家，直接傳遞當前的 alpha, beta 邊界
        // 如果換玩家，傳入翻轉變號後的 [-beta, -alpha]
        if (same) {
            raw_score = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
        } else {
            raw_score = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
        }

        int score = same ? raw_score : -raw_score;
        delete next;

        // 更新此節點最佳分數
        best_score = score > best_score ? score : best_score;

        // 更新 Alpha 下界
        if (score > alpha) {
            alpha = score;
        }

        // Beta 截斷 (剪枝)：當前情況已經優於對手會容忍的範圍，對手絕不會走這步
        if (alpha >= beta) {
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * AlphaBeta — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult AlphaBeta::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    ABParams p = ABParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    // 根節點的 Alpha/Beta 初始界限 (用足夠大的數值代表負無窮大與正無窮大)
    int alpha = -1000000;
    int beta  =  1000000;

    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    result.best_move = state->legal_actions[0];

    for(auto& action : state->legal_actions){
        State *next = state->next_state(action);
        bool same = next->same_player_as_parent();

        history.push(state->hash());
        
        int raw_score;
        if (same) {
            raw_score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
        } else {
            raw_score = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
        }
        
        history.pop(state->hash());

        int score = same ? raw_score : -raw_score;
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }  

        // 根節點即時更新 alpha，之後的其他分支才能用最新的條件剪枝
        if (score > alpha) {
            alpha = score;
        }

        move_index++;
    }

    result.score = best_score;
    return result;
} 


/*============================================================
 * AlphaBeta — default_params / param_defs
 *============================================================*/
ParamMap AlphaBeta::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> AlphaBeta::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
